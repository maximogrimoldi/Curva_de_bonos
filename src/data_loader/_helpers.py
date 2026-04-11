"""Helpers compartidos entre los módulos de data_loader."""

import calendar
from datetime import date

SETTLEMENT_DATE = date(2026, 4, 10)


def sinking(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    pct: float,
    day: int = 9,
    frequency: int = 2,
) -> list:
    """
    Genera una lista de (date, pct) para un sinking fund regular.

    Parámetros
    ----------
    start_year/month : primer pago de amortización
    end_year/month   : último pago de amortización
    pct              : fracción del face_value a amortizar en cada cuota
    day              : día del mes de los pagos (31 → usa el último día del mes)
    frequency        : pagos por año (2=semestral, 4=trimestral)

    Ejemplo
    -------
    sinking(2025, 1, 2029, 7, 0.10)         → 10 cuotas semestrales de 10%
    sinking(2024, 6, 2033, 12, 0.05, day=31) → 20 cuotas semestrales EOM (Jun-30/Dic-31)
    """
    step = 12 // frequency
    result = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        last = calendar.monthrange(y, m)[1]
        result.append((date(y, m, min(day, last)), pct))
        m += step
        if m > 12:
            m -= 12
            y += 1
    return result
