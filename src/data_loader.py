"""
Schedules aproximados de los Globales USD (reestructuración 2020) y ONs de ejemplo.
Para uso en producción verificar contra el Indenture oficial (SEC EDGAR) o Bloomberg.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from .bond import Bond

SETTLEMENT_DATE = date(2026, 4, 10)


def _semiannual_dates(start_year: int, start_month: int, end_year: int, end_month: int) -> list:
    """Fechas de pago semi-anuales (día 9) entre dos fechas (Jan y Jul)."""
    dates = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        dates.append(date(year, month, 9))
        month += 6
        if month > 12:
            month -= 12
            year += 1
    return dates


def _build_gd29(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2029, 7)
    amort_dates = _semiannual_dates(2023, 1, 2029, 7)[:8]
    amort_map = {d: 1 / 8 for d in amort_dates}
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [
        {"start_date": date(2020, 9, 4), "end_date": date(2022, 1, 9), "rate": 0.00125},
        {"start_date": date(2022, 1, 9), "end_date": date(2024, 1, 9), "rate": 0.00750},
        {"start_date": date(2024, 1, 9), "end_date": date(2029, 7, 9), "rate": 0.01750},
    ]
    return Bond("GD29", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows))


def _build_gd30(settlement: date) -> Bond:
    all_dates = _semiannual_dates(2021, 1, 2030, 7)
    amort_dates = _semiannual_dates(2026, 7, 2030, 7)
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


def load_sample_ons(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """ONs corporativas USD de emisores argentinos reales."""

    # YPF S.A. — 9.00% Notes due 2029 (bullet)
    # Serie emitida en julio 2022; pagos semestrales el 9 de ene/jul.
    ypf_amort = [
        {"date": d, "amort_pct": (1.0 if d == date(2029, 7, 9) else 0.0)}
        for d in _semiannual_dates(2022, 7, 2029, 7)
    ]
    ypf29 = Bond("YPF29", 100.0, settlement,
                 pd.DataFrame(ypf_amort),
                 pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2029, 7, 9), "rate": 0.0900}]))

    # Telecom Argentina — 8.50% Notes due 2031 (bullet)
    # Pagos semestrales el 9 de ene/jul; vto enero 2031.
    teco_amort = [
        {"date": d, "amort_pct": (1.0 if d == date(2031, 1, 9) else 0.0)}
        for d in _semiannual_dates(2021, 7, 2031, 1)
    ]
    teco31 = Bond("TECO31", 100.0, settlement,
                  pd.DataFrame(teco_amort),
                  pd.DataFrame([{"start_date": date(2021, 1, 1), "end_date": date(2031, 1, 9), "rate": 0.0850}]))

    return {
        "YPF29":  (ypf29,  86.00),
        "TECO31": (teco31, 78.00),
    }
