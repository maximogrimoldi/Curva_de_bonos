"""
Bonos soberanos Dollar Linked (capital ajustado por TC oficial, COM A3500) y ONs corporativas DL.
Precios como % del valor técnico (capital indexado). Schedules aproximados.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _semiannual_dates, _may_nov_dates


# ── Bonos soberanos Dollar Linked ─────────────────────────────────────────────

def _build_tv26d(settlement: date) -> Bond:
    """TV26D: Dollar Linked 0.50% anual, bullet Nov 9 2026."""
    all_dates = _may_nov_dates(2024, 2026)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2026, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2023, 1, 1), "end_date": date(2026, 11, 9), "rate": 0.0050}]
    return Bond("TV26D", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_DL")


def _build_tv27(settlement: date) -> Bond:
    """TV27: Dollar Linked 0.50% anual, bullet Nov 9 2027."""
    all_dates = _may_nov_dates(2024, 2027)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2027, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2023, 1, 1), "end_date": date(2027, 11, 9), "rate": 0.0050}]
    return Bond("TV27", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_DL")


def _build_tv28(settlement: date) -> Bond:
    """TV28: Dollar Linked 1.00% anual, bullet Nov 9 2028."""
    all_dates = _may_nov_dates(2024, 2028)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2028, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2023, 1, 1), "end_date": date(2028, 11, 9), "rate": 0.0100}]
    return Bond("TV28", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_DL")


def _build_tv30(settlement: date) -> Bond:
    """TV30: Dollar Linked 1.50% anual, bullet Nov 9 2030."""
    all_dates = _may_nov_dates(2024, 2030)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2030, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2023, 1, 1), "end_date": date(2030, 11, 9), "rate": 0.0150}]
    return Bond("TV30", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_DL")


def load_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Bonos del Tesoro Dollar Linked: TV26D, TV27, TV28, TV30."""
    builders = {
        "TV26D": _build_tv26d,
        "TV27":  _build_tv27,
        "TV28":  _build_tv28,
        "TV30":  _build_tv30,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


# ── ONs corporativas Dollar Linked ────────────────────────────────────────────

def load_sample_ons_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """ONs corporativas Dollar Linked: VISTAD28 (Vista Oil) y PAMPDL29 (Pampa Energía)."""

    # VISTAD28: Vista Oil & Gas — ON DL 2.00%, bullet Abr 9 2028
    vista_dates = _semiannual_dates(2024, 4, 2028, 4)
    vista_amort = [{"date": d, "amort_pct": (1.0 if d == date(2028, 4, 9) else 0.0)} for d in vista_dates]
    vistad28 = Bond(
        "VISTAD28", 100.0, settlement,
        pd.DataFrame(vista_amort),
        pd.DataFrame([{"start_date": date(2024, 1, 1), "end_date": date(2028, 4, 9), "rate": 0.0200}]),
        currency="ARS_DL",
    )

    # PAMPDL29: Pampa Energía — ON DL 1.50%, bullet Oct 9 2029
    pampa_dates = _semiannual_dates(2024, 4, 2029, 10)
    pampa_amort = [{"date": d, "amort_pct": (1.0 if d == date(2029, 10, 9) else 0.0)} for d in pampa_dates]
    pampdl29 = Bond(
        "PAMPDL29", 100.0, settlement,
        pd.DataFrame(pampa_amort),
        pd.DataFrame([{"start_date": date(2024, 1, 1), "end_date": date(2029, 10, 9), "rate": 0.0150}]),
        currency="ARS_DL",
    )

    return {
        "VISTAD28": (vistad28, 97.00),
        "PAMPDL29": (pampdl29, 94.00),
    }
