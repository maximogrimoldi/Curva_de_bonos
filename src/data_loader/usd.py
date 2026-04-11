"""
Bonos soberanos USD — Reestructuración 2020 (Ley Nueva York) y ONs corporativas USD.

Convenciones globales:
- Day Count:  30/360
- Calendar:   AR (Buenos Aires) — Business Day Convention: Following Unadjusted
- Pagos:      9 de enero y 9 de julio (Globales)
- Precios:    Dirty Price como % del face value nominal

Fuente: Indentures SEC EDGAR / Prospecto de Reestructuración Sep-2020.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _semiannual_dates, _quarterly_dates


# ── Globales USD ──────────────────────────────────────────────────────────────

def _build_gd29(settlement: date) -> Bond:
    """
    GD29 — Global 2029 | Cupón actual: 1.00% | Vto: 09-Jul-2029
    Amortización: 10 cuotas de 10% c/u, semestral desde Ene-2025.
    Residual Apr-2026: 70% (pagadas: Ene-25, Jul-25, Ene-26).
    """
    all_dates   = _semiannual_dates(2021, 1, 2029, 7)
    amort_dates = _semiannual_dates(2025, 1, 2029, 7)  # 10 fechas × 10% = 100%
    amort_map   = {d: 0.10 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2029, 7, 9), "rate": 0.0100}]
    return Bond("GD29", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


def _build_gd30(settlement: date) -> Bond:
    """
    GD30 — Global 2030 | Cupón actual: 0.75% | Vto: 09-Jul-2030
    Amortización: 12 cuotas semianuales desde Jul-2024 (4%×2, 8%×10) + residual 12% en vto.
    Residual Apr-2026: ~76% (pagadas: Jul-24, Ene-25, Jul-25, Ene-26).
    """
    all_dates  = _semiannual_dates(2021, 1, 2030, 7)
    # 12 cuotas de sinking fund: primeras 2 al 4%, siguientes 10 al 8% = 88%
    sf_dates   = _semiannual_dates(2024, 7, 2030, 1)   # 12 fechas (Jul-24 → Ene-30)
    amort_map  = {d: (0.04 if i < 2 else 0.08) for i, d in enumerate(sf_dates)}
    amort_map[date(2030, 7, 9)] = 0.12                 # residual 12% en vencimiento
    amort_rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2030, 7, 9), "rate": 0.0075}]
    return Bond("GD30", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


def _build_gd35(settlement: date) -> Bond:
    """
    GD35 — Global 2035 | Cupón actual: 4.75% | Vto: 09-Jul-2035
    Amortización: 10 cuotas de 10% c/u, semestral desde Ene-2031.
    Residual Apr-2026: 100%.
    """
    all_dates   = _semiannual_dates(2021, 1, 2035, 7)
    amort_dates = _semiannual_dates(2031, 1, 2035, 7)  # 10 fechas × 10% = 100%
    amort_map   = {d: 0.10 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2035, 7, 9), "rate": 0.0475}]
    return Bond("GD35", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


def _build_gd38(settlement: date) -> Bond:
    """
    GD38 — Global 2038 | Cupón actual: 5.00% | Vto: 09-Ene-2038
    Amortización: 20 cuotas de 5% c/u, semestral desde Jul-2028.
    Residual Apr-2026: 100%.
    """
    all_dates   = _semiannual_dates(2021, 1, 2038, 1)
    amort_dates = _semiannual_dates(2028, 7, 2038, 1)  # 20 fechas × 5% = 100%
    amort_map   = {d: 0.05 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2038, 1, 9), "rate": 0.0500}]
    return Bond("GD38", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


def _build_gd41(settlement: date) -> Bond:
    """
    GD41 — Global 2041 | Cupón actual: 4.875% | Vto: 09-Jul-2041
    Amortización: 28 cuotas de 1/28 c/u, semestral desde Ene-2028.
    Residual Apr-2026: 100%.
    """
    all_dates   = _semiannual_dates(2021, 1, 2041, 7)
    amort_dates = _semiannual_dates(2028, 1, 2041, 7)  # 28 fechas × 1/28 = 100%
    amort_map   = {d: round(1 / 28, 10) for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2041, 7, 9), "rate": 0.04875}]
    return Bond("GD41", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


def _build_gd46(settlement: date) -> Bond:
    """
    GD46 — Global 2046 | Cupón actual: 5.00% | Vto: 09-Jul-2046
    Amortización: 40 cuotas de 2.5% c/u, semestral desde Ene-2027.
    Residual Apr-2026: 100%.
    """
    all_dates   = _semiannual_dates(2021, 1, 2046, 7)
    amort_dates = _semiannual_dates(2027, 1, 2046, 7)  # 40 fechas × 2.5% = 100%
    amort_map   = {d: 0.025 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 9, 4), "end_date": date(2046, 7, 9), "rate": 0.0500}]
    return Bond("GD46", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                day_count="30/360")


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
    """
    ONs corporativas USD:
    - YMCQO  (YPF 9.00%,  sinking fund trimestral, vto Sep-2029)
    - TEC3O  (Telecom 8.50%, sinking fund semestral, vto Jun-2031)
    """

    # ── YMCQO: YPF S.A. 9.00% Notes due Sep-30-2029 ──────────────────────────
    # Pagos trimestrales: Mar-30, Jun-30, Sep-30, Dic-30.
    # Sinking fund con cuotas crecientes desde Sep-2025.
    # Aproximación: 16 cuotas iguales de 6.25% c/u (Sep-25 → Jun-29).
    ypf_all   = _quarterly_dates(2022, 9, 2029, 9)
    ypf_sf    = _quarterly_dates(2025, 9, 2029, 6)     # 16 cuotas × 6.25% = 100% (Sep-25 → Jun-29)
    ypf_amap  = {d: round(1 / 16, 10) for d in ypf_sf}
    ypf_amort = [{"date": d, "amort_pct": ypf_amap.get(d, 0.0)} for d in ypf_all]
    ymcqo = Bond(
        "YMCQO", 100.0, settlement,
        pd.DataFrame(ypf_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2029, 9, 30), "rate": 0.0900}]),
        frequency=4,
        day_count="30/360",
    )

    # ── TEC3O: Telecom Argentina 8.50% Notes due Jun-16-2031 ─────────────────
    # Cupones semianuales el 16 de Jun y 16 de Dic desde emisión.
    # Sinking fund semestral: 5 cuotas de 20% desde Jun-2029.
    teco_all = [
        date(2021, 6, 16), date(2021, 12, 16),
        date(2022, 6, 16), date(2022, 12, 16),
        date(2023, 6, 16), date(2023, 12, 16),
        date(2024, 6, 16), date(2024, 12, 16),
        date(2025, 6, 16), date(2025, 12, 16),
        date(2026, 6, 16), date(2026, 12, 16),
        date(2027, 6, 16), date(2027, 12, 16),
        date(2028, 6, 16), date(2028, 12, 16),
        date(2029, 6, 16), date(2029, 12, 16),
        date(2030, 6, 16), date(2030, 12, 16),
        date(2031, 6, 16),
    ]
    teco_sf   = [date(2029, 6, 16), date(2029, 12, 16),
                 date(2030, 6, 16), date(2030, 12, 16), date(2031, 6, 16)]
    teco_amap = {d: 0.20 for d in teco_sf}
    teco_amort = [{"date": d, "amort_pct": teco_amap.get(d, 0.0)} for d in teco_all]
    tec3o = Bond(
        "TEC3O", 100.0, settlement,
        pd.DataFrame(teco_amort),
        pd.DataFrame([{"start_date": date(2021, 1, 1), "end_date": date(2031, 6, 16), "rate": 0.0850}]),
        day_count="30/360",
    )

    return {
        "YMCQO": (ymcqo, 86.00),
        "TEC3O": (tec3o, 78.00),
    }
