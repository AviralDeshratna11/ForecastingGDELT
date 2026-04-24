"""
utils/kelly_criterion.py
------------------------
Kelly Criterion for optimal bet sizing in prediction markets.

Full Kelly:   f* = (b·p - q) / b
where:
  f* = fraction of bankroll to wager
  b  = net odds (profit per unit at risk)
  p  = estimated true probability
  q  = 1 - p

For prediction markets (binary contracts priced at [0,1]):
  b  = (1 - market_price) / market_price   [for YES contracts]
  f* = p - market_price   (simplified for equal-payout markets)

Fractional Kelly (recommended): f_actual = f* × fraction
where fraction ∈ [0.1, 0.5] for safety margin.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class KellyResult:
    full_kelly: float
    fractional_kelly: float
    fraction: float
    edge: float          # true_prob - market_price
    odds_ratio: float    # b
    expected_growth: float


class KellyCriterion:
    """
    Computes optimal bet fractions for prediction market contracts.

    Usage:
        kelly = KellyCriterion()
        f = kelly.compute(true_prob=0.65, market_price=0.50)
        # f = 0.15  → bet 15% of bankroll on YES

        result = kelly.full_analysis(true_prob=0.65, market_price=0.50, fraction=0.25)
    """

    def compute(self, true_prob: float, market_price: float) -> float:
        """
        Simplified full Kelly for binary prediction markets.

        f* = (p - q_market) / (1 - q_market)
           = p - market_price     [for fair market]

        Returns the full Kelly fraction (can be negative → bet NO).
        """
        p = float(np.clip(true_prob, 0.001, 0.999))
        m = float(np.clip(market_price, 0.001, 0.999))

        # YES bet: profit = (1-m)/m per unit; loss = 1 unit
        b_yes = (1 - m) / m
        f_yes = (b_yes * p - (1 - p)) / b_yes

        # NO bet: profit = m/(1-m) per unit
        b_no  = m / (1 - m)
        f_no  = (b_no * (1 - p) - p) / b_no

        # Pick the direction with positive edge
        if f_yes > 0:
            return float(np.clip(f_yes, 0.0, 1.0))
        elif f_no > 0:
            return float(np.clip(-f_no, -1.0, 0.0))  # negative = bet NO
        else:
            return 0.0  # no edge

    def full_analysis(
        self,
        true_prob: float,
        market_price: float,
        fraction: float = 0.25,
        bankroll: float = 1.0,
    ) -> KellyResult:
        """Full analysis with growth rate calculation."""
        p = float(np.clip(true_prob, 0.001, 0.999))
        m = float(np.clip(market_price, 0.001, 0.999))

        b = (1 - m) / m   # YES odds
        f_star = (b * p - (1 - p)) / b

        f_actual = float(np.clip(f_star * fraction, -1.0, 1.0))

        edge = p - m

        # Expected log growth: E[log(1 + f*outcome)]
        if f_star > 0:
            growth = p * np.log(1 + b * abs(f_actual)) + (1 - p) * np.log(1 - abs(f_actual))
        else:
            growth = 0.0

        return KellyResult(
            full_kelly=round(float(f_star), 4),
            fractional_kelly=round(float(f_actual), 4),
            fraction=fraction,
            edge=round(float(edge), 4),
            odds_ratio=round(float(b), 4),
            expected_growth=round(float(growth), 6),
        )

    def optimal_fraction_search(
        self,
        true_prob: float,
        market_price: float,
        grid_size: int = 100,
    ) -> float:
        """
        Numerically search for the fraction that maximises expected log growth.
        Useful for validating the closed-form solution.
        """
        p = float(np.clip(true_prob, 0.001, 0.999))
        m = float(np.clip(market_price, 0.001, 0.999))
        b = (1 - m) / m

        fractions = np.linspace(0, 0.99, grid_size)
        growths = p * np.log(1 + b * fractions) + (1 - p) * np.log(1 - fractions)

        best_idx = np.argmax(growths)
        return float(fractions[best_idx])
