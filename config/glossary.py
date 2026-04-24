"""
config/glossary.py
------------------
Definitions of every metric and term used in the GDELT forecasting system.
Injected into the LLM analyst prompt so it interprets numbers correctly.
"""

GDELT_GLOSSARY = """
=== METRIC DEFINITIONS (read carefully before interpreting signals) ===

GDELT DATA METRICS:
  AvgTone           : Average emotional tone of all news articles. Scale: -100 to +100.
                      Negative = conflict/negativity. Positive = cooperation/positivity.
                      Typical range for geopolitical events: -5 to +2.
                      Below -3 = high conflict environment. Above +2 = cooperative environment.

  GoldsteinScale    : Theoretical conflict/cooperation potential of each event type.
                      Scale: -10 (most destabilising) to +10 (most stabilising).
                      Negative values = conflict-generating events dominate.
                      Positive values = cooperation-generating events dominate.
                      E.g. -8 = military assault, -5 = sanctions, +6 = diplomatic agreement.

  NumMentions       : Total number of times an event was mentioned across all articles.
                      Higher = more media attention = more significant event.

  NumSources        : Number of distinct news sources covering the event.
                      Higher = broader geographic/editorial coverage = more credible signal.

  NumArticles       : Unique articles mentioning the event.
                      Spike in NumArticles = breaking news or escalating situation.

CAMEO QuadClass (Event Classification):
  Q1 Verbal Cooperation   : Diplomatic statements, agreements in words. Low intensity.
  Q2 Material Cooperation : Physical aid, summits, treaties signed. Moderate-high positive.
  Q3 Verbal Conflict      : Threats, accusations, denunciations. Moderate tension.
  Q4 Material Conflict    : Physical violence, military action, sanctions. High intensity.
  High Q4 % = active physical conflict. High Q3 % = escalating diplomatic tension.

GKG (Global Knowledge Graph) METRICS:
  positive_score    : % of article text with positive sentiment (LIWC-based).
  negative_score    : % of article text with negative sentiment.
  polarity          : negative_score - positive_score. Higher = more negative coverage.
  activity_density  : Volume of named entities per article. Higher = more complex events.

STATISTICAL METRICS:
  Z-score           : How many standard deviations the current value is from the 60-day mean.
                      |z| < 1.0 = normal variation. 1-2 = elevated. >2.0 = statistically significant.
                      >3.0 = extreme outlier — major event likely occurred.
                      Negative z on AvgTone = significantly more negative than baseline.
                      Positive z on AvgTone = significantly more positive than baseline.

  Inflection Point  : A date where |z-score| crossed the 2.0 threshold in either direction.
                      Represents a statistically significant shift in media coverage or sentiment.
                      Multiple inflections in short period = volatile/rapidly evolving situation.

  Rolling Z-score   : Computed over a 60-day window. Resets the baseline every 60 days.
                      Designed to detect recent anomalies relative to the recent past, not all-time history.

FORECAST METRICS:
  Trend Direction   : "up" = signal improving (less conflict/more cooperation). 
                      "down" = signal deteriorating. "flat" = no clear direction.
  Trend Strength    : 0 to 1. How strong the directional movement is. >0.5 = strong trend.
  Momentum          : Rate of change of the trend slope. 
                      Positive = acceleration. Negative = deceleration/reversal forming.
  Uncertainty       : Average width of the 80% confidence interval. 
                      Higher = model less certain. >2.0 = very uncertain forecast.

PROBABILITY CALIBRATION:
  Raw Probability   : LLM's initial estimate based on signals.
  Calibrated Prob   : After Platt scaling to remove LLM over/under-confidence bias.
  Extremized Prob   : After power-transform (alpha=2.5) — pushes away from 50% toward 0 or 1.
                      More useful for prediction markets where decisive signals exist.
  Kelly Fraction    : Optimal bankroll fraction to bet if treating this as a prediction market.
                      Negative Kelly = bet NO. Positive = bet YES. 0 = no edge.
  Brier Score       : Accuracy measure. 0=perfect, 0.25=random, 1=worst. Lower is better.
"""
