"""
main.py — GDELT Multi-Agent Predictive Forecasting System
Usage:
    python main.py --event "Russia Ukraine war escalation" --days 90
    python main.py --event "oil price spike" --days 60 --market-price 0.45
"""

import json
import logging
import os
import sys
import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich.columns import Columns

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG
from agents.orchestrator import ForecastingOrchestrator
from utils.visualizer import ForecastVisualizer

app = Console(width=120)
cli = typer.Typer(add_completion=False)


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
def predict(
    event: str = typer.Option(..., "--event", "-e"),
    description: str = typer.Option(None, "--description", "-d"),
    days: int = typer.Option(90, "--days"),
    market_price: float = typer.Option(0.5, "--market-price"),
    output_format: str = typer.Option("rich", "--format", "-f"),
    visualise: bool = typer.Option(True, "--visualise/--no-visualise"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Run the GDELT multi-agent event outcome prediction pipeline."""
    _setup_logging(log_level)

    app.print(Panel.fit(
        "[bold cyan]GDELT Multi-Agent Forecasting System[/bold cyan]\n"
        "[dim]GDELT · TSFM · Agentic AI · Bayesian Calibration[/dim]",
        border_style="blue",
    ))

    app.print(f"\n[yellow]▶ Query:[/yellow] [bold]{event}[/bold]")
    orchestrator = ForecastingOrchestrator.build(CONFIG)

    with app.status("[bold green]Running multi-agent pipeline...[/bold green]"):
        result = orchestrator.run(
            event_query=event,
            event_description=description or event,
            lookback_days=days,
            market_price=market_price,
        )

    final        = result.get("final_output", {})
    consequences = result.get("consequences", [])
    scenarios    = result.get("scenarios", [])
    indicators   = result.get("key_indicators", [])

    if output_format == "json":
        out = {**final, "consequences": consequences, "scenarios": scenarios, "key_indicators": indicators}
        print(json.dumps(out, indent=2, default=str))
    elif output_format == "minimal":
        prob = final.get("final_probability", 0.0)
        rec  = final.get("recommendation", "NO_TRADE")
        print(f"PROBABILITY={prob:.4f}  RECOMMENDATION={rec}")
    else:
        _rich_output(final, event, consequences, scenarios, indicators)

    if visualise:
        signals  = result.get("gdelt_signals")
        forecast = result.get("tsfm_forecast")
        if signals and forecast:
            app.print("\n[yellow]▶ Generating charts...[/yellow]")
            viz = ForecastVisualizer(output_dir=CONFIG.output_dir)
            paths = viz.plot_all(signals, forecast, final, event)
            for p in paths:
                app.print(f"  [green]✓[/green] {p}")

    os.makedirs(CONFIG.output_dir, exist_ok=True)
    ts   = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = event[:30].replace(" ", "_").lower()
    out  = {**final, "consequences": consequences, "scenarios": scenarios, "key_indicators": indicators}
    json_path = os.path.join(CONFIG.output_dir, f"{ts}_{slug}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    app.print(f"\n[dim]Saved → {json_path}[/dim]")


def _rich_output(final: dict, event: str, consequences: list, scenarios: list, indicators: list):
    prob   = final.get("final_probability", 0.0)
    rec    = final.get("recommendation", final.get("recommended_position", "NO_TRADE"))
    kelly  = final.get("kelly_fraction", 0.0)
    reasoning = final.get("reasoning", "")

    prob_color = "green" if prob > 0.55 else "red" if prob < 0.45 else "yellow"
    rec_color  = {"YES": "green", "NO": "red", "NO_TRADE": "yellow"}.get(rec, "white")

    # ── 1. VERDICT PANEL ──────────────────────────────────────────────
    app.print()
    app.print(Rule("[bold cyan]FORECAST VERDICT[/bold cyan]", style="cyan"))
    app.print(Panel(
        f"[bold white]Event:[/bold white]  {event}\n\n"
        f"[bold white]Probability:[/bold white]  [{prob_color} bold]{prob:.1%}[/{prob_color} bold]   "
        f"[dim](Raw: {final.get('raw_probability',0):.1%}  →  Calibrated: {final.get('calibrated_probability',0):.1%}  →  Extremized: {final.get('extremized_probability',0):.1%})[/dim]\n\n"
        f"[bold white]Signal:[/bold white]  Trend [{final.get('trend_analysis',{}).get('overall_trend','—')}]  "
        f"Z-score [{final.get('trend_analysis',{}).get('latest_z_score',0):.2f}]  "
        f"Inflections [{final.get('trend_analysis',{}).get('n_inflections',0)}]\n\n"
        f"[bold white]Decision:[/bold white]  [{rec_color} bold]{rec}[/{rec_color} bold]   "
        f"Kelly: [{rec_color}]{kelly:+.1%}[/{rec_color}]",
        border_style=prob_color,
        padding=(1, 2),
    ))

    # ── 2. AGENT REASONING ────────────────────────────────────────────
    if reasoning:
        app.print(Panel(
            reasoning.strip(),
            title="[bold yellow]Why This Probability[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        ))

    # ── 3. FUTURE CONSEQUENCES ────────────────────────────────────────
    app.print()
    app.print(Rule("[bold magenta]What Happens Next — Predicted Consequences[/bold magenta]", style="magenta"))

    if consequences:
        for i, c in enumerate(consequences, 1):
            sev      = str(c.get("severity", "medium")).lower()
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
            sev_col  = {"high": "red", "medium": "yellow", "low": "green"}.get(sev, "white")
            regions  = ", ".join(c.get("regions", [])) if isinstance(c.get("regions"), list) else str(c.get("regions", ""))
            prob_occ = c.get("probability_if_event_occurs", 0)

            app.print(Panel(
                f"{sev_icon}  [{sev_col} bold]{sev.upper()} SEVERITY[/{sev_col} bold]   "
                f"[dim]Probability if event occurs: {prob_occ:.0%}[/dim]\n\n"
                f"[white bold]{c.get('consequence', '')}[/white bold]\n\n"
                f"[dim]⏱  {c.get('timeframe', '')}   |   📍 {regions}[/dim]",
                title=f"[magenta]Consequence {i}[/magenta]",
                border_style=sev_col,
                padding=(0, 2),
            ))
    else:
        app.print("[dim]No consequence data returned — check LLM connection[/dim]")

    # ── 4. SCENARIO ANALYSIS ──────────────────────────────────────────
    app.print()
    app.print(Rule("[bold blue]Scenario Analysis — Next 30 Days[/bold blue]", style="blue"))

    if scenarios:
        sc_colors = {"Accelerated": "green", "Base Case": "yellow", "Reversal": "red"}
        sc_icons  = {"Accelerated": "🚀", "Base Case": "📊", "Reversal": "🔄"}
        for sc in scenarios:
            label    = str(sc.get("label", ""))
            color    = sc_colors.get(label, "white")
            icon     = sc_icons.get(label, "•")
            prob_val = float(sc.get("probability", 0))

            app.print(Panel(
                f"{icon}  [{color} bold]{label}[/{color} bold]   "
                f"[{color}]{prob_val:.0%} probability[/{color}]\n\n"
                f"[white]{sc.get('description', '')}[/white]\n\n"
                f"[dim]🔑 Trigger: {sc.get('key_trigger', '')}   |   "
                f"📅 Expected: {sc.get('expected_date', '')}[/dim]",
                border_style=color,
                padding=(0, 2),
            ))
    else:
        app.print("[dim]No scenario data returned — check LLM connection[/dim]")

    # ── 5. KEY INDICATORS ─────────────────────────────────────────────
    if indicators:
        app.print()
        app.print(Rule("[bold green]Watch These Indicators[/bold green]", style="green"))
        for i, ind in enumerate(indicators, 1):
            app.print(f"  [green bold]{i}.[/green bold] {ind}")

    # ── 6. TECHNICAL DETAIL (collapsed at bottom) ─────────────────────
    app.print()
    app.print(Rule("[dim]Technical Detail[/dim]", style="dim"))
    detail_table = Table(show_header=False, box=None, padding=(0, 2))
    detail_table.add_column("Metric", style="dim", width=24)
    detail_table.add_column("Value")
    ta = final.get("trend_analysis", {})
    rows = [
        ("Forecast model",      ta.get("forecast_model", "—")),
        ("Forecast slope",      f"{ta.get('forecast_slope', 0):+.5f}"),
        ("Trend strength",      f"{ta.get('forecast_strength', 0):.3f}"),
        ("Momentum",            f"{ta.get('forecast_momentum', 0):+.4f}"),
        ("Uncertainty",         f"{ta.get('forecast_uncertainty', 0):.4f}"),
        ("Brier score",         f"{final.get('evaluation_score', 0):.4f}  (0=perfect, 0.25=random)"),
        ("Reflection cycles",   str(final.get("iterations", 1))),
        ("Significant events",  str(ta.get("n_significant", 0))),
    ]
    for k, v in rows:
        detail_table.add_row(k, v)
    app.print(detail_table)
    app.print()


if __name__ == "__main__":
    cli()
