from __future__ import annotations

import math


def decimal(value: object, places: int = 2) -> str:
    """Formato de lectura usado en las aplicaciones Elephant Data Labs."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    text = f"{number:,.{places}f}"
    integer, _, fraction = text.partition(".")
    integer = integer.replace(",", ".")
    return f"{integer},{fraction}" if places else integer


def percent(value: object, places: int = 1) -> str:
    try:
        return decimal(float(value) * 100, places) + "%"
    except (TypeError, ValueError):
        return "—"
