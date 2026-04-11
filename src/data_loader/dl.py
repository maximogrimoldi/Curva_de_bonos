"""
Bonos soberanos Dollar Linked (capital ajustado por TC oficial, COM A3500) y ONs corporativas DL.

Convenciones globales:
- Day Count:  Actual/Actual
- Precios:    Dirty Price como % del valor técnico (capital ajustado por TC oficial)

Fuente: Prospectos CNV / indentures de emisión.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _jun_dec_dates


# ── Bonos soberanos Dollar Linked ─────────────────────────────────────────────

def _build_tv27(settlement: date) -> Bond:
    """
    TV27 — Dollar Linked 2027 | Cupón: 0.50% | Vto: 23-Nov-2027
    Amortización: bullet (100% en vencimiento).
    Residual Apr-2026: 100%.
    """
    # Cupones semianuales el 23 de May y 23 de Nov
    all_dates = [
        date(2024, 5, 23), date(2024, 11, 23),
        date(2025, 5, 23), date(2025, 11, 23),
        date(2026, 5, 23), date(2026, 11, 23),
        date(2027, 5, 23), date(2027, 11, 23),
    ]
    amort_rows  = [{"date": d, "amort_pct": (1.0 if d == date(2027, 11, 23) else 0.0)}
                   for d in all_dates]
    coupon_rows = [{"start_date": date(2023, 1, 1), "end_date": date(2027, 11, 23), "rate": 0.0050}]
    return Bond("TV27", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_DL", day_count="Actual/Actual")


def _build_tv28(settlement: date) -> Bond:
    """
    TV28 — Dollar Linked 2028 | Cupón: 0.50% | Vto: 30-Jun-2028
    Amortización: bullet (100% en vencimiento).
    Residual Apr-2026: 100%.
    """
    # Cupones semianuales el 30 de Jun y 31 de Dic
    all_dates = _jun_dec_dates(2025, 2028)
    all_dates = [d for d in all_dates if d <= date(2028, 6, 30)]
    amort_rows  = [{"date": d, "amort_pct": (1.0 if d == date(2028, 6, 30) else 0.0)}
                   for d in all_dates]
    coupon_rows = [{"start_date": date(2024, 1, 1), "end_date": date(2028, 6, 30), "rate": 0.0050}]
    return Bond("TV28", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_DL", day_count="Actual/Actual")


def _build_tv30(settlement: date) -> Bond:
    """
    TV30 — Dollar Linked 2030 | Cupón: 0.75% | Vto: 30-Nov-2030
    Amortización: bullet (100% en vencimiento).
    Residual Apr-2026: 100%.
    """
    # Cupones semianuales el 31 de May y 30 de Nov
    all_dates = [
        date(2025, 5, 31), date(2025, 11, 30),
        date(2026, 5, 31), date(2026, 11, 30),
        date(2027, 5, 31), date(2027, 11, 30),
        date(2028, 5, 31), date(2028, 11, 30),
        date(2029, 5, 31), date(2029, 11, 30),
        date(2030, 5, 31), date(2030, 11, 30),
    ]
    amort_rows  = [{"date": d, "amort_pct": (1.0 if d == date(2030, 11, 30) else 0.0)}
                   for d in all_dates]
    coupon_rows = [{"start_date": date(2024, 1, 1), "end_date": date(2030, 11, 30), "rate": 0.0075}]
    return Bond("TV30", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_DL", day_count="Actual/Actual")


def load_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Bonos del Tesoro Dollar Linked: TV27, TV28, TV30."""
    builders = {
        "TV27": _build_tv27,
        "TV28": _build_tv28,
        "TV30": _build_tv30,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


EXAMPLE_MARKET_PRICES_DL: Dict[str, float] = {
    "TV27": 96.00,
    "TV28": 93.00,
    "TV30": 88.00,
}


# ── ONs corporativas Dollar Linked ────────────────────────────────────────────

def load_sample_ons_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas Dollar Linked:
    - VISTAD28  (Vista Oil & Gas 3.50%, bullet Ago-15-2028)
    - PAMPDL29  (Pampa Energía 5.00%, bullet Oct-10-2029)
    """

    # ── VISTAD28: Vista Oil & Gas — ON DL 3.50%, bullet Aug-15-2028 ──────────
    # Cupones semianuales el 15 de Feb y 15 de Ago
    vista_all = [
        date(2025, 2, 15), date(2025, 8, 15),
        date(2026, 2, 15), date(2026, 8, 15),
        date(2027, 2, 15), date(2027, 8, 15),
        date(2028, 2, 15), date(2028, 8, 15),
    ]
    vista_amort = [{"date": d, "amort_pct": (1.0 if d == date(2028, 8, 15) else 0.0)}
                   for d in vista_all]
    vistad28 = Bond(
        "VISTAD28", 100.0, settlement,
        pd.DataFrame(vista_amort),
        pd.DataFrame([{"start_date": date(2024, 1, 1), "end_date": date(2028, 8, 15), "rate": 0.0350}]),
        currency="ARS_DL",
        day_count="Actual/Actual",
    )

    # ── PAMPDL29: Pampa Energía — ON DL 5.00%, bullet Oct-10-2029 ────────────
    # Cupones semianuales el 10 de Abr y 10 de Oct
    pampa_all = [
        date(2025, 4, 10), date(2025, 10, 10),
        date(2026, 4, 10), date(2026, 10, 10),
        date(2027, 4, 10), date(2027, 10, 10),
        date(2028, 4, 10), date(2028, 10, 10),
        date(2029, 4, 10), date(2029, 10, 10),
    ]
    pampa_amort = [{"date": d, "amort_pct": (1.0 if d == date(2029, 10, 10) else 0.0)}
                   for d in pampa_all]
    pampdl29 = Bond(
        "PAMPDL29", 100.0, settlement,
        pd.DataFrame(pampa_amort),
        pd.DataFrame([{"start_date": date(2024, 1, 1), "end_date": date(2029, 10, 10), "rate": 0.0500}]),
        currency="ARS_DL",
        day_count="Actual/Actual",
    )

    return {
        "VISTAD28": (vistad28, 97.00),
        "PAMPDL29": (pampdl29, 94.00),
    }
