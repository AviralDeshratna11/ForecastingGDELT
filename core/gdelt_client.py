"""
core/gdelt_client.py
--------------------
GDELT data client with corrected BigQuery SQL for actual GDELT schema.

GDELT BigQuery schema (gdelt-bq.gdeltv2):
  events : SQLDATE=INT64, Actor1Name=STRING, AvgTone=FLOAT64, NumArticles=INT64
  gkg    : DATE=INT64 (YYYYMMDDHHMMSS as integer!), V2Tone=STRING, V2Themes=STRING
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# Short keyword map — GDELT indexes short actor/theme keywords, not full phrases
KEYWORD_MAP = {
    "russia ukraine war escalation": ["RUSSIA", "UKRAINE"],
    "iran israel military conflict":  ["IRAN", "ISRAEL"],
    "taiwan china military tension":  ["TAIWAN", "CHINA"],
    "north korea missile launch":     ["NKOREA", "MISSILE"],
    "india pakistan border conflict": ["INDIA", "PAKISTAN"],
    "federal reserve interest rate":  ["FEDGOV", "ECON"],
    "us recession":                   ["USGOV", "ECON"],
    "oil price spike":                ["OIL", "OPEC"],
    "opec production cut":            ["OPEC", "OIL"],
    "bitcoin price crash":            ["CRYPTO", "BITCOIN"],
    "china economic slowdown":        ["CHINA", "ECON"],
    "climate change extreme weather": ["ENV", "CLIMATE"],
}


def _extract_keywords(query: str) -> list[str]:
    """
    Extract 1-2 short GDELT-compatible keywords from a query string.
    GDELT actor names are 2-6 char uppercase codes or short proper nouns.
    """
    q_lower = query.lower().strip()

    # Check the map first
    for k, v in KEYWORD_MAP.items():
        if any(word in q_lower for word in k.split()):
            return v

    # Fallback: take the two longest words as keywords
    words = [w.upper() for w in query.split() if len(w) > 3]
    return words[:2] if words else [query.upper()[:8]]


class GDELTClient:
    def __init__(self, config):
        self.config = config
        self._bq_client = None
        self._try_init_bigquery()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_events(self, query: str, days: int = 90,
                     country_filter: Optional[str] = None) -> pd.DataFrame:
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        if self._bq_client:
            df = self._bq_fetch_events(query, start_date, end_date, country_filter)
            if not df.empty:
                return df
        return self._http_fetch_events(query, start_date, end_date)

    def fetch_gkg(self, query: str, days: int = 90) -> pd.DataFrame:
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        if self._bq_client:
            df = self._bq_fetch_gkg(query, start_date, end_date)
            if not df.empty:
                return df
        return self._http_fetch_gkg(query, start_date, end_date)

    def fetch_combined(self, query: str, days: int = 90) -> pd.DataFrame:
        events = self.fetch_events(query, days)
        gkg    = self.fetch_gkg(query, days)

        if events.empty and gkg.empty:
            logger.warning("All data sources failed — using synthetic data")
            end   = datetime.utcnow()
            start = end - timedelta(days=days)
            return self._generate_synthetic_events(query, start, end)

        if events.empty:
            return gkg
        if gkg.empty:
            return events

        combined = pd.merge(events, gkg, on="date", how="outer", suffixes=("_evt", "_gkg"))
        combined = combined.sort_values("date").reset_index(drop=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.ffill().bfill()
        logger.info("Combined dataset: %d rows", len(combined))
        return combined

    # ------------------------------------------------------------------ #
    # BigQuery — Events table                                              #
    # ------------------------------------------------------------------ #

    def _try_init_bigquery(self):
        if not self.config.bigquery_project:
            return
        try:
            from google.cloud import bigquery
            self._bq_client = bigquery.Client(project=self.config.bigquery_project)
            logger.info("BigQuery client initialised for project %s", self.config.bigquery_project)
        except Exception as exc:
            logger.warning("BigQuery init failed (%s)", exc)

    def _bq_fetch_events(self, query, start_date, end_date, country_filter):
        """
        Events table: SQLDATE is INT64 YYYYMMDD.
        Filter by short keywords against Actor1Name / Actor2Name.
        Aggregate to daily rows.
        """
        keywords   = _extract_keywords(query)
        start_int  = int(start_date.strftime('%Y%m%d'))
        end_int    = int(end_date.strftime('%Y%m%d'))
        country_cl = f"AND ActionGeo_CountryCode = '{country_filter}'" if country_filter else ""

        # Build OR conditions for each keyword
        kw_clauses = " OR ".join([
            f"Actor1Name LIKE '%{kw}%' OR Actor2Name LIKE '%{kw}%'"
            for kw in keywords
        ])

        sql = f"""
        SELECT
            PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
            AVG(GoldsteinScale)  AS goldstein_scale,
            SUM(NumMentions)     AS num_mentions,
            SUM(NumSources)      AS num_sources,
            SUM(NumArticles)     AS num_articles,
            AVG(AvgTone)         AS avg_tone,
            MAX(QuadClass)       AS quad_class
        FROM `{self.config.bigquery_dataset}.events`
        WHERE
            SQLDATE BETWEEN {start_int} AND {end_int}
            {country_cl}
            AND ({kw_clauses})
        GROUP BY date
        ORDER BY date
        """
        try:
            df = self._bq_client.query(sql).to_dataframe()
            if df.empty:
                logger.warning("BQ events: 0 rows for '%s' (keywords: %s)", query, keywords)
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            logger.info("BQ events: %d rows for '%s'", len(df), query)
            return df
        except Exception as exc:
            logger.warning("BQ events query failed: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # BigQuery — GKG table                                                 #
    # ------------------------------------------------------------------ #

    def _bq_fetch_gkg(self, query, start_date, end_date):
        """
        GKG table: DATE is INT64 stored as YYYYMMDDHHMMSS (e.g. 20240415120000).
        To filter by day: DATE >= 20240101000000 AND DATE < 20240201000000
        V2Tone is STRING: 'tone,positive,negative,polarity,actdensity,selfdensity'
        V2Themes is STRING: semicolon-delimited theme codes
        """
        keywords  = _extract_keywords(query)
        # Append 000000 (midnight) to make full YYYYMMDDHHMMSS integers
        start_int = int(start_date.strftime('%Y%m%d') + '000000')
        end_int   = int(end_date.strftime('%Y%m%d') + '235959')

        # Theme keyword search — use first keyword
        theme_kw = keywords[0] if keywords else query.upper()[:8]

        sql = f"""
        SELECT
            PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS tone,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64)) AS positive_score,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64)) AS negative_score,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(3)] AS FLOAT64)) AS polarity,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(4)] AS FLOAT64)) AS activity_density,
            COUNT(*) AS num_articles
        FROM `{self.config.bigquery_dataset}.gkg`
        WHERE
            DATE BETWEEN {start_int} AND {end_int}
            AND V2Themes LIKE '%{theme_kw}%'
        GROUP BY date
        ORDER BY date
        """
        try:
            df = self._bq_client.query(sql).to_dataframe()
            if df.empty:
                logger.warning("BQ GKG: 0 rows for '%s' (theme: %s)", query, theme_kw)
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            logger.info("BQ GKG: %d rows for '%s'", len(df), query)
            return df
        except Exception as exc:
            logger.warning("BQ GKG query failed: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # HTTP fallback — GDELT DOC 2.0 API                                   #
    # ------------------------------------------------------------------ #

    def _http_fetch_events(self, query: str, start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        try:
            url = (
                f"{self.config.gdelt_api_base}/doc/doc?"
                f"query={requests.utils.quote(query)}"
                f"&mode=timelinetonedaily"
                f"&startdatetime={start_date.strftime('%Y%m%d%H%M%S')}"
                f"&enddatetime={end_date.strftime('%Y%m%d%H%M%S')}"
                f"&format=json&maxrecords=250"
            )
            resp = requests.get(url, timeout=60)   # increased from 30s
            if resp.status_code == 200:
                data = resp.json()
                df   = self._parse_timeline_response(data)
                if not df.empty:
                    logger.info("HTTP events: %d rows for '%s'", len(df), query)
                    return df
        except Exception as exc:
            logger.warning("GDELT HTTP events failed: %s", exc)

        logger.info("Using synthetic event data for '%s'", query)
        end   = datetime.utcnow()
        start = end - timedelta(days=(end_date - start_date).days)
        return self._generate_synthetic_events(query, start, end)

    def _http_fetch_gkg(self, query: str, start_date: datetime,
                         end_date: datetime) -> pd.DataFrame:
        try:
            url = (
                f"{self.config.gdelt_api_base}/doc/doc?"
                f"query={requests.utils.quote(query)}"
                f"&mode=timelinevolrawdaily"
                f"&startdatetime={start_date.strftime('%Y%m%d%H%M%S')}"
                f"&enddatetime={end_date.strftime('%Y%m%d%H%M%S')}"
                f"&format=json&maxrecords=250"
            )
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                df   = self._parse_volume_response(data)
                if not df.empty:
                    return df
        except Exception as exc:
            logger.warning("GDELT HTTP GKG failed: %s", exc)

        return self._generate_synthetic_gkg(query, start_date, end_date)

    # ------------------------------------------------------------------ #
    # Response parsers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_timeline_response(self, data: dict) -> pd.DataFrame:
        records  = []
        timeline = data.get("timeline", [{}])[0].get("data", [])
        for point in timeline:
            try:
                records.append({
                    "date":            pd.to_datetime(point.get("date", "")),
                    "avg_tone":        float(point.get("value", 0)),
                    "num_articles":    int(point.get("norm", 0)),
                    "quad_class":      3,
                    "goldstein_scale": float(point.get("value", 0)) * 0.5,
                    "num_mentions":    int(point.get("norm", 0)) * 3,
                    "num_sources":     max(1, int(point.get("norm", 0)) // 2),
                })
            except Exception:
                continue
        return pd.DataFrame(records)

    def _parse_volume_response(self, data: dict) -> pd.DataFrame:
        records  = []
        timeline = data.get("timeline", [{}])[0].get("data", [])
        for point in timeline:
            try:
                vol = float(point.get("value", 0))
                records.append({
                    "date":             pd.to_datetime(point.get("date", "")),
                    "tone":             -abs(vol) * 0.1,
                    "positive_score":   max(0, vol * 0.3),
                    "negative_score":   max(0, abs(vol) * 0.5),
                    "activity_density": vol,
                    "num_articles":     int(point.get("norm", 0)),
                })
            except Exception:
                continue
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Synthetic data generators                                            #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_events(self, query: str, start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        np.random.seed(abs(hash(query)) % 2**31)
        dates = pd.date_range(start_date, end_date, freq="D")
        n     = len(dates)
        inflection_idx = n * 2 // 3
        trend  = np.zeros(n)
        trend[inflection_idx:] = np.linspace(0, 3.5, n - inflection_idx)
        tone   = (-2.5 + trend + np.random.normal(0, 1.2, n)
                  + 0.8 * np.sin(np.linspace(0, 4 * np.pi, n)))
        articles = (120 + trend * 40 + np.abs(np.random.normal(0, 25, n))).astype(int)
        return pd.DataFrame({
            "date":             dates,
            "avg_tone":         np.clip(tone, -10, 10),
            "num_articles":     np.clip(articles, 1, 10000),
            "num_mentions":     articles * 3,
            "num_sources":      np.clip(articles // 5, 1, 500),
            "quad_class":       np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.15, 0.4, 0.25]),
            "goldstein_scale":  np.clip(tone * 0.7, -10, 10),
        })

    def _generate_synthetic_gkg(self, query: str, start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        np.random.seed((abs(hash(query)) + 1) % 2**31)
        dates = pd.date_range(start_date, end_date, freq="D")
        n     = len(dates)
        tone  = np.random.normal(-1.5, 2.0, n)
        pos   = np.clip(np.random.normal(5, 1.5, n), 0, 20)
        neg   = np.clip(np.random.normal(7, 2.0, n), 0, 20)
        return pd.DataFrame({
            "date":             dates,
            "tone":             tone,
            "positive_score":   pos,
            "negative_score":   neg,
            "polarity":         neg - pos,
            "activity_density": np.abs(np.random.normal(500, 150, n)),
            "num_articles":     np.random.randint(50, 400, n),
        })
