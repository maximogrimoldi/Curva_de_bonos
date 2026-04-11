"""
Bonos soberanos Dollar Linked (capital ajustado por TC oficial, COM A3500)
y ONs corporativas DL.

Convenciones:
- Day Count : Actual/Actual
- Precios   : Dirty Price como % del valor técnico (capital ajustado por TC oficial)

Fuente: Prospectos CNV / indentures de emisión.
"""

from datetime import date
from typing import Dict, Tuple

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE


# ── Bonos soberanos Dollar Linked ─────────────────────────────────────────────

def load_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Bonos del Tesoro Dollar Linked: TV27, TV28, TV30 (bullets)."""
    return {
        "TV27": Bond("TV27", 100.0, settlement, date(2027, 11, 23),
                     coupon=0.0050, currency="ARS_DL", day_count="Actual/Actual"),
        "TV28": Bond("TV28", 100.0, settlement, date(2028, 6, 30),
                     coupon=0.0050, currency="ARS_DL", day_count="Actual/Actual"),
        "TV30": Bond("TV30", 100.0, settlement, date(2030, 11, 30),
                     coupon=0.0075, currency="ARS_DL", day_count="Actual/Actual"),
    }


EXAMPLE_MARKET_PRICES_DL: Dict[str, float] = {
    "TV27": 92.15, "TV28": 90.14, "TV30": 85.20,
}


# ── ONs corporativas Dollar Linked ────────────────────────────────────────────

def load_sample_ons_dollar_linked(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas Dollar Linked:
    - VISTAD28  (Vista Oil & Gas 3.50%, bullet Ago-15-2028)
    - PAMPDL29  (Pampa Energía 5.00%, bullet Oct-10-2029)
    """
    vistad28 = Bond("VISTAD28", 100.0, settlement, date(2028, 8, 15),
                    coupon=0.0350, currency="ARS_DL", day_count="Actual/Actual")

    pampdl29 = Bond("PAMPDL29", 100.0, settlement, date(2029, 10, 10),
                    coupon=0.0500, currency="ARS_DL", day_count="Actual/Actual")

    return {
        "VISTAD28": (vistad28, 97.00),
        "PAMPDL29": (pampdl29, 94.00),
    }
