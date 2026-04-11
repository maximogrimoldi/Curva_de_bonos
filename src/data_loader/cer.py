"""
Bonos soberanos ajustados por CER (tasa real ARS) y ONs corporativas ajustadas por UVA.

Convenciones globales:
- Day Count:  Actual/Actual (convención estándar para instrumentos indexados en AR)
- Pagos:      9 de mayo y 9 de noviembre (BONCER) | 30 de junio y 31 de diciembre (DICP, PARP)
- Precios:    Dirty Price como % del valor técnico (capital ajustado por CER)

Fuente: Prospectos CNV / BCRA / indentures de emisión.
"""

from datetime import date
from typing import Dict, Tuple

import pandas as pd

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, _semiannual_dates, _jun_dec_dates


# ── Bonos soberanos CER ───────────────────────────────────────────────────────

def _build_tx26(settlement: date) -> Bond:
    """
    TX26 — BONCER 2026 | Cupón real: 2.00% | Vto: 09-Nov-2026
    Amortización: 5 cuotas de 20% c/u, semestral desde Nov-2024.
    Residual Apr-2026: ~40% (pagadas: Nov-24, May-25, Nov-25).
    """
    all_dates   = _semiannual_dates(2021, 5, 2026, 11)
    amort_dates = _semiannual_dates(2024, 11, 2026, 11)   # 5 fechas × 20% = 100%
    amort_map   = {d: 0.20 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2021, 1, 1), "end_date": date(2026, 11, 9), "rate": 0.0200}]
    return Bond("TX26", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_tx28(settlement: date) -> Bond:
    """
    TX28 — BONCER 2028 | Cupón real: 2.25% | Vto: 09-Nov-2028
    Amortización: 10 cuotas de 10% c/u, semestral desde May-2024.
    Residual Apr-2026: 60% (pagadas: May-24, Nov-24, May-25, Nov-25).
    """
    all_dates   = _semiannual_dates(2022, 5, 2028, 11)
    amort_dates = _semiannual_dates(2024, 5, 2028, 11)    # 10 fechas × 10% = 100%
    amort_map   = {d: 0.10 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2022, 1, 1), "end_date": date(2028, 11, 9), "rate": 0.0225}]
    return Bond("TX28", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_tx30(settlement: date) -> Bond:
    """
    TX30 — BONCER 2030 | Cupón real: 2.50% | Vto: 09-Nov-2030
    Amortización: bullet (100% en vencimiento).
    Residual Apr-2026: 100%.
    """
    all_dates  = _semiannual_dates(2022, 5, 2030, 11)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2030, 11, 9) else 0.0)}
                  for d in all_dates]
    coupon_rows = [{"start_date": date(2022, 1, 1), "end_date": date(2030, 11, 9), "rate": 0.0250}]
    return Bond("TX30", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_dicp(settlement: date) -> Bond:
    """
    DICP — Discount CER | Cupón real: 5.83% | Vto: 31-Dic-2033
    Amortización: 20 cuotas de 5% c/u, semestral desde Jun-2024.
    Residual Apr-2026: 80% (pagadas: Jun-24, Dic-24, Jun-25, Dic-25).
    """
    all_dates   = _jun_dec_dates(2021, 2033)
    amort_dates = _jun_dec_dates(2024, 2033)               # 20 fechas × 5% = 100%
    amort_map   = {d: 0.05 for d in amort_dates}
    amort_rows  = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 1, 1), "end_date": date(2033, 12, 31), "rate": 0.0583}]
    return Bond("DICP", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_parp(settlement: date) -> Bond:
    """
    PARP — Par CER | Cupón real: 1.83% | Vto: 31-Dic-2038
    Amortización: bullet (100% en vencimiento).
    Residual Apr-2026: 100%.
    """
    all_dates  = _jun_dec_dates(2021, 2038)
    amort_rows = [{"date": d, "amort_pct": (1.0 if d == date(2038, 12, 31) else 0.0)}
                  for d in all_dates]
    coupon_rows = [{"start_date": date(2020, 1, 1), "end_date": date(2038, 12, 31), "rate": 0.0183}]
    return Bond("PARP", 100.0, settlement,
                pd.DataFrame(amort_rows), pd.DataFrame(coupon_rows),
                currency="ARS_CER", day_count="Actual/Actual")


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


EXAMPLE_MARKET_PRICES_CER: Dict[str, float] = {
    "TX26": 96.65,
    "TX28": 90.57,
    "TX30": 88.79,
    "DICP": 90.33,
    "PARP": 80.79,
}


# ── ONs corporativas UVA ──────────────────────────────────────────────────────

def load_sample_ons_cer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas ajustadas por UVA:
    - TLCJO  (Telecom Serie J 2.50%, bullet Jun-30-2026)
    - IRCPO  (IRSA Propiedades 4.00%, 3 cuotas: 33%+33%+34%, vto Mar-25-2028)
    """

    # ── TLCJO: Telecom Argentina Serie J — ON UVA 2.50%, bullet Jun-30-2026 ───
    tlcjo_all = _jun_dec_dates(2022, 2026)          # Jun-30 y Dic-31 de 2022 a 2026
    tlcjo_all = [d for d in tlcjo_all if d <= date(2026, 6, 30)]
    tlcjo_amort = [{"date": d, "amort_pct": (1.0 if d == date(2026, 6, 30) else 0.0)}
                   for d in tlcjo_all]
    tlcjo = Bond(
        "TLCJO", 100.0, settlement,
        pd.DataFrame(tlcjo_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2026, 6, 30), "rate": 0.0250}]),
        currency="ARS_UVA",
        day_count="Actual/Actual",
    )

    # ── IRCPO: IRSA Propiedades Comerciales — ON UVA 4.00%, 3 cuotas, vto Mar-25-2028 ──
    # 3 cuotas anuales: 33% en Mar-2026, 33% en Mar-2027, 34% en Mar-2028.
    # Cupones semianuales el 25 de Sep y 25 de Mar.
    ircpo_all = [
        date(2022, 9, 25), date(2023, 3, 25), date(2023, 9, 25),
        date(2024, 3, 25), date(2024, 9, 25), date(2025, 3, 25),
        date(2025, 9, 25), date(2026, 3, 25), date(2026, 9, 25),
        date(2027, 3, 25), date(2027, 9, 25), date(2028, 3, 25),
    ]
    ircpo_amort_map = {
        date(2026, 3, 25): 0.33,
        date(2027, 3, 25): 0.33,
        date(2028, 3, 25): 0.34,
    }
    ircpo_amort = [{"date": d, "amort_pct": ircpo_amort_map.get(d, 0.0)} for d in ircpo_all]
    ircpo = Bond(
        "IRCPO", 100.0, settlement,
        pd.DataFrame(ircpo_amort),
        pd.DataFrame([{"start_date": date(2022, 1, 1), "end_date": date(2028, 3, 25), "rate": 0.0400}]),
        currency="ARS_UVA",
        day_count="Actual/Actual",
    )

    return {
        "TLCJO": (tlcjo, 98.00),
        "IRCPO": (ircpo, 94.00),
    }
