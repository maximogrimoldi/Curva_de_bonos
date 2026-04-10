"""
Bonos soberanos USD (Globales, reestructuración 2020) y ONs corporativas USD.
Schedules aproximados — verificar contra Indenture oficial (SEC EDGAR) o Bloomberg.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _semiannual_dates


# ── Globales USD ──────────────────────────────────────────────────────────────

def _build_gd29(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2029, 7)
    # Amortización en 6 cuotas iguales desde Ene 2027 hasta Jul 2029 (todas futuras a Apr 2026)
    amort_dates = _semiannual_dates(2027, 1, 2029, 7)  # 6 fechas
    amort_map = {d: 1 / 6 for d in amort_dates}
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [
        {"start_date": date(2020, 9, 4), "end_date": date(2022, 1, 9), "rate": 0.00125},
        {"start_date": date(2022, 1, 9), "end_date": date(2024, 1, 9), "rate": 0.00750},
        {"start_date": date(2024, 1, 9), "end_date": date(2029, 7, 9), "rate": 0.01750},
    ]
    return Bond("GD29", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd30(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2030, 7)
    # 5 cuotas iguales Jul 2026 – Jul 2028; [:5] evita pagos fantasma
    amort_dates = _semiannual_dates(2026, 7, 2030, 7)[:5]
    amort_map = {d: 1 / 5 for d in amort_dates}
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [
        {"start_date": date(2020, 9, 4), "end_date": date(2021, 7, 9), "rate": 0.00500},
        {"start_date": date(2021, 7, 9), "end_date": date(2022, 7, 9), "rate": 0.01000},
        {"start_date": date(2022, 7, 9), "end_date": date(2024, 1, 9), "rate": 0.01750},
        {"start_date": date(2024, 1, 9), "end_date": date(2030, 7, 9), "rate": 0.02000},
    ]
    return Bond("GD30", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd35(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2035, 7)
    amort_dates = _semiannual_dates(2033, 1, 2035, 7)[:4]
    amort_map = {d: 0.25 for d in amort_dates}
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2035, 7, 9), "rate": 0.03625}]
    return Bond("GD35", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd38(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2038, 1)
    amort_dates = _semiannual_dates(2034, 1, 2038, 1)[:6]
    amort_map = {d: 1 / 6 for d in amort_dates}
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [
        {"start_date": date(2020, 9, 4), "end_date": date(2023, 1, 9), "rate": 0.02000},
        {"start_date": date(2023, 1, 9), "end_date": date(2038, 1, 9), "rate": 0.03500},
    ]
    return Bond("GD38", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd41(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2041, 7)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2041, 7, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2041, 7, 9), "rate": 0.04125}]
    return Bond("GD41", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd46(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2046, 7)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2046, 7, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2046, 7, 9), "rate": 0.03625}]
    return Bond("GD46", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def load_globales(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Globales USD reestructuración 2020: GD29, GD30, GD35, GD38, GD41, GD46."""
    builders = {
        "GD29": _build_gd29,
        "GD30": _build_gd30,
        "GD35": _build_gd35,
        "GD38": _build_gd38,
        "GD41": _build_gd41,
        "GD46": _build_gd46,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


EXAMPLE_MARKET_PRICES: Dict[str, float] = {
    "GD29": 71.50,
    "GD30": 68.00,
    "GD35": 62.00,
    "GD38": 60.00,
    "GD41": 58.50,
    "GD46": 56.00,
}


# ── ONs corporativas USD ──────────────────────────────────────────────────────

def load_sample_ons(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """ONs corporativas USD: YPF29 (9.00%) y TECO31 (8.50%)."""

    # YPF S.A. — 9.00% Notes due 2029 (bullet), pagos semestrales ene/jul
    ypf_amort = [
        {"date": d, "amort_pct": (1.0 if d == date(2029, 7, 9) else 0.0)}
        for d in _semiannual_dates(2022, 7, 2029, 7)
    ]
    ypf29 = Bond(
        "YPF29", 100.0, settlement,
        pd.DataFrame(ypf_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2029, 7, 9), "rate": 0.0900}]),
    )

    # Telecom Argentina — 8.50% Notes due 2031 (bullet), pagos semestrales ene/jul
    teco_amort = [
        {"date": d, "amort_pct": (1.0 if d == date(2031, 1, 9) else 0.0)}
        for d in _semiannual_dates(2021, 7, 2031, 1)
    ]
    teco31 = Bond(
        "TECO31", 100.0, settlement,
        pd.DataFrame(teco_amort),
        pd.DataFrame([{"start_date": date(2021, 1, 1), "end_date": date(2031, 1, 9), "rate": 0.0850}]),
    )

    return {
        "YPF29":  (ypf29,  86.00),
        "TECO31": (teco31, 78.00),
    }
