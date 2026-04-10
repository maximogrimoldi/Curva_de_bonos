"""
Bonos soberanos ajustados por CER (tasa real ARS) y ONs corporativas ajustadas por UVA.
Precios como % del valor técnico (capital indexado). Schedules aproximados.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _semiannual_dates, _may_nov_dates, _jun_dec_dates


# ── Bonos soberanos CER ───────────────────────────────────────────────────────

def _build_tx26(settlement: date) -> Bond:
    """TX26: BONCER 1.45% real anual, bullet Nov 9 2026."""
    all_dates = _may_nov_dates(2021, 2026)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2026, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 1, 1), "end_date": date(2026, 11, 9), "rate": 0.0145}]
    return Bond("TX26", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_CER")


def _build_tx28(settlement: date) -> Bond:
    """TX28: BONCER 2.00% real anual, bullet Nov 9 2028."""
    all_dates = _may_nov_dates(2022, 2028)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2028, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2021, 1, 1), "end_date": date(2028, 11, 9), "rate": 0.0200}]
    return Bond("TX28", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_CER")


def _build_tx30(settlement: date) -> Bond:
    """TX30: BONCER 2.50% real anual, bullet Nov 9 2030."""
    all_dates = _may_nov_dates(2022, 2030)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2030, 11, 9) else 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2021, 1, 1), "end_date": date(2030, 11, 9), "rate": 0.0250}]
    return Bond("TX30", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_CER")


def _build_dicp(settlement: date) -> Bond:
    """DICP: Discount CER 5.83% real anual, sinking fund semestral Jun/Dic 2024-2034 (5% por cuota)."""
    all_dates = _jun_dec_dates(2024, 2034)
    # 20 cuotas de 5% c/u desde Dic 2024 hasta Jun 2034; Jun 2024 es sólo cupón
    amort_rows = [
        {"date": d, "amort_pct": (0.05 if d >= date(2024, 12, 31) else 0.0)}
        for d in all_dates
    ]
    coupon_rows = [{"start_date": date(2020, 1, 1), "end_date": date(2034, 12, 31), "rate": 0.0583}]
    return Bond("DICP", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_CER")


def _build_parp(settlement: date) -> Bond:
    """PARP: Par CER step-up 1.33%→2.67%, bullet Dic 31 2038."""
    all_dates = _jun_dec_dates(2021, 2038)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2038, 12, 31) else 0.0)} for d in all_dates]
    coupon_rows = [
        {"start_date": date(2020, 1, 1),   "end_date": date(2024, 12, 31), "rate": 0.0133},
        {"start_date": date(2024, 12, 31), "end_date": date(2038, 12, 31), "rate": 0.0267},
    ]
    return Bond("PARP", 100.0, settlement, pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows), currency="ARS_CER")


def load_boncer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Bonos del Tesoro ajustados por CER: TX26, TX28, TX30, DICP, PARP."""
    builders = {
        "TX26": _build_tx26,
        "TX28": _build_tx28,
        "TX30": _build_tx30,
        "DICP": _build_dicp,
        "PARP": _build_parp,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


# ── ONs corporativas UVA ──────────────────────────────────────────────────────

def load_sample_ons_cer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """ONs corporativas ajustadas por UVA: TLCJO (Telecom) e IRCPO (IRSA)."""

    # TLCJO: Telecom Argentina Serie J — ON UVA 3.50% real, bullet Jun 9 2027
    tlcjo_dates = _semiannual_dates(2022, 6, 2027, 6)
    tlcjo_amort = [{"date": d, "amort_pct": (1.0 if d == date(2027, 6, 9) else 0.0)} for d in tlcjo_dates]
    tlcjo = Bond(
        "TLCJO", 100.0, settlement,
        pd.DataFrame(tlcjo_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2027, 6, 9), "rate": 0.0350}]),
        currency="ARS_UVA",
    )

    # IRCPO: IRSA Propiedades Comerciales — ON UVA 4.00% real, bullet Mar 9 2028
    ircpo_dates = _semiannual_dates(2022, 3, 2028, 3)
    ircpo_amort = [{"date": d, "amort_pct": (1.0 if d == date(2028, 3, 9) else 0.0)} for d in ircpo_dates]
    ircpo = Bond(
        "IRCPO", 100.0, settlement,
        pd.DataFrame(ircpo_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2028, 3, 9), "rate": 0.0400}]),
        currency="ARS_UVA",
    )

    return {
        "TLCJO": (tlcjo, 96.00),
        "IRCPO": (ircpo, 94.00),
    }
