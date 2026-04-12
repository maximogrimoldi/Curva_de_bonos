"""
Bonos soberanos USD — Reestructuración 2020 (Ley Nueva York) y ONs corporativas USD.

Convenciones:
- Day Count : 30/360
- Pagos     : 9 de enero y 9 de julio (Globales)
- Precios   : Dirty Price como % del face value nominal

Fuente: Indentures SEC EDGAR / Prospecto de Reestructuración Sep-2020.
"""

from datetime import date
from typing import Dict, Tuple

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, sinking


# ── Globales USD ──────────────────────────────────────────────────────────────

def _build_gd29(settlement: date) -> Bond:
    """GD29 — Global 2029 | 1.00% | 10 × 10% desde Ene-2025 | Residual Apr-26: 70%"""
    return Bond("GD29", 100.0, settlement, date(2029, 7, 9),
                coupon=0.0100, amort=sinking(2025, 1, 2029, 7, 0.10),
                day_count="30/360")


def _build_gd30(settlement: date) -> Bond:
    """GD30 — Global 2030 | 0.75% | 4%×2 + 8%×10 + 12% bullet | Residual Apr-26: 76%"""
    amort = (sinking(2024, 7, 2025, 1, 0.04) +
             sinking(2025, 7, 2030, 1, 0.08) +
             [(date(2030, 7, 9), 0.12)])
    return Bond("GD30", 100.0, settlement, date(2030, 7, 9),
                coupon=0.0075, amort=amort, day_count="30/360")


def _build_gd35(settlement: date) -> Bond:
    """GD35 — Global 2035 | 4.75% | 10 × 10% desde Ene-2031 | Residual Apr-26: 100%"""
    return Bond("GD35", 100.0, settlement, date(2035, 7, 9),
                coupon=0.0475, amort=sinking(2031, 1, 2035, 7, 0.10),
                day_count="30/360")


def _build_gd38(settlement: date) -> Bond:
    """GD38 — Global 2038 | 5.00% | 20 × 5% desde Jul-2028 | Residual Apr-26: 100%"""
    return Bond("GD38", 100.0, settlement, date(2038, 1, 9),
                coupon=0.0500, amort=sinking(2028, 7, 2038, 1, 0.05),
                day_count="30/360")


def _build_gd41(settlement: date) -> Bond:
    """GD41 — Global 2041 | 4.875% | 28 × 1/28 desde Ene-2028 | Residual Apr-26: 100%"""
    return Bond("GD41", 100.0, settlement, date(2041, 7, 9),
                coupon=0.04875, amort=sinking(2028, 1, 2041, 7, round(1/28, 10)),
                day_count="30/360")


def _build_gd46(settlement: date) -> Bond:
    """GD46 — Global 2046 | 5.00% | 40 × 2.5% desde Ene-2027 | Residual Apr-26: 100%"""
    return Bond("GD46", 100.0, settlement, date(2046, 7, 9),
                coupon=0.0500, amort=sinking(2027, 1, 2046, 7, 0.025),
                day_count="30/360")


def load_globales(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Globales USD reestructuración 2020: GD29, GD30, GD35, GD38, GD41, GD46."""
    builders = {
        "GD29": _build_gd29, "GD30": _build_gd30, "GD35": _build_gd35,
        "GD38": _build_gd38, "GD41": _build_gd41, "GD46": _build_gd46,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


EXAMPLE_MARKET_PRICES: Dict[str, float] = {
    "GD29": 65.49, "GD30": 64.64, "GD35": 79.90,
    "GD38": 83.10, "GD41": 74.17, "GD46": 70.99,
}


# ── ONs corporativas USD ──────────────────────────────────────────────────────

def load_sample_ons(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas USD:
    - YMCQOD  (YPF 9.00%, trimestral, 16 × 6.25% desde Sep-2025, vto Sep-2029)
    - TLC10D  (Telecom 8.50%, semestral, 5 × 20% desde Jun-2029, vto Jun-2031)
    """
    ymcqod = Bond("YMCQOD", 100.0, settlement, date(2029, 9, 30),
                  coupon=0.0900, frequency=4,
                  amort=sinking(2025, 9, 2029, 6, round(1/16, 10), day=30, frequency=4),
                  day_count="30/360")

    tlc10d = Bond("TLC10D", 100.0, settlement, date(2031, 6, 16),
                  coupon=0.0850,
                  amort=sinking(2029, 6, 2031, 6, 0.20, day=16),
                  day_count="30/360")

    return {
        "YMCQOD": (ymcqod, 102.00),
        "TLC10D": (tlc10d, 103.30),
    }
