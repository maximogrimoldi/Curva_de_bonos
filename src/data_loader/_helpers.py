"""Helpers compartidos entre los módulos de data_loader."""

from datetime import date

SETTLEMENT_DATE = date(2026, 4, 10)


def _semiannual_dates(start_year: int, start_month: int, end_year: int, end_month: int) -> list:
    """Fechas de pago semi-anuales (día 9) entre dos fechas."""
    dates = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        dates.append(date(year, month, 9))
        month += 6
        if month > 12:
            month -= 12
            year += 1
    return dates


def _may_nov_dates(start_year: int, end_year: int) -> list:
    """Fechas semianuales el día 9 de mayo y noviembre."""
    dates = []
    for y in range(start_year, end_year + 1):
        dates.append(date(y, 5, 9))
        dates.append(date(y, 11, 9))
    return sorted(dates)


def _jun_dec_dates(start_year: int, end_year: int) -> list:
    """Fechas semianuales el 30 de junio y 31 de diciembre."""
    dates = []
    for y in range(start_year, end_year + 1):
        dates.append(date(y, 6, 30))
        dates.append(date(y, 12, 31))
    return sorted(dates)
