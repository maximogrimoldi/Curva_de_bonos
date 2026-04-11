"""
Bonos soberanos ajustados por CER (tasa real ARS) y ONs corporativas ajustadas por UVA.

Convenciones:
- Day Count : Actual/Actual
- Pagos     : 9 de mayo y noviembre (TX), 30 Jun / 31 Dic (DICP/PARP)
- Precios   : Dirty Price como % del valor técnico (capital ajustado por CER)

Fuente: Prospectos CNV / BCRA / indentures de emisión.
"""

from datetime import date
from typing import Dict, Tuple

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, sinking


# ── Bonos soberanos CER ───────────────────────────────────────────────────────

def _build_tx26(settlement: date) -> Bond:
    """TX26 — BONCER 2026 | 2.00% real | 5 × 20% desde Nov-2024 | Residual Apr-26: 40%"""
    return Bond("TX26", 100.0, settlement, date(2026, 11, 9),
                coupon=0.0200, amort=sinking(2024, 11, 2026, 11, 0.20),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_tx28(settlement: date) -> Bond:
    """TX28 — BONCER 2028 | 2.25% real | 10 × 10% desde May-2024 | Residual Apr-26: 60%"""
    return Bond("TX28", 100.0, settlement, date(2028, 11, 9),
                coupon=0.0225, amort=sinking(2024, 5, 2028, 11, 0.10),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_tx30(settlement: date) -> Bond:
    """TX30 — BONCER 2030 | 2.50% real | bullet | Residual Apr-26: 100%"""
    return Bond("TX30", 100.0, settlement, date(2030, 11, 9),
                coupon=0.0250,
                currency="ARS_CER", day_count="Actual/Actual")


def _build_dicp(settlement: date) -> Bond:
    """DICP — Discount CER | 5.83% real | 20 × 5% desde Jun-2024 (EOM) | Residual Apr-26: 80%"""
    return Bond("DICP", 100.0, settlement, date(2033, 12, 31),
                coupon=0.0583, amort=sinking(2024, 6, 2033, 12, 0.05, day=31),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_parp(settlement: date) -> Bond:
    """PARP — Par CER | 1.83% real | bullet | Residual Apr-26: 100%"""
    return Bond("PARP", 100.0, settlement, date(2038, 12, 31),
                coupon=0.0183,
                currency="ARS_CER", day_count="Actual/Actual")


def load_boncer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Bonos del Tesoro ajustados por CER: TX26, TX28, TX30, DICP, PARP."""
    builders = {
        "TX26": _build_tx26, "TX28": _build_tx28, "TX30": _build_tx30,
        "DICP": _build_dicp, "PARP": _build_parp,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


EXAMPLE_MARKET_PRICES_CER: Dict[str, float] = {
    "TX26": 39.49, "TX28": 57.05, "TX30": 88.79, "DICP": 81.74, "PARP": 72.79,
}


# ── ONs corporativas UVA ──────────────────────────────────────────────────────

def load_sample_ons_cer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas ajustadas por UVA:
    - TLCJO  (Telecom Serie J 2.50%, bullet Jun-30-2026)
    - IRCPO  (IRSA Propiedades 4.00%, 3 cuotas anuales 33%+33%+34%, vto Mar-25-2028)
    """
    tlcjo = Bond("TLCJO", 100.0, settlement, date(2026, 6, 30),
                 coupon=0.0250,
                 currency="ARS_UVA", day_count="Actual/Actual")

    ircpo = Bond("IRCPO", 100.0, settlement, date(2028, 3, 25),
                 coupon=0.0400,
                 amort=[(date(2026, 3, 25), 0.33),
                        (date(2027, 3, 25), 0.33),
                        (date(2028, 3, 25), 0.34)],
                 currency="ARS_UVA", day_count="Actual/Actual")

    return {
        "TLCJO": (tlcjo, 98.00),
        "IRCPO": (ircpo, 94.00),
    }
