from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from resources.config import MISSING_MARKER
from resources.config import UNIT_COLS


def _numeric_to_float(num):
    """
    Parse numreical value as number or as the average of a range
    """
    try:
        return float(num)
    except Exception:
        text = re.sub(r"[−–—‐-]", "-", str(num))
        pattern = r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?-[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
        if re.match(pattern, text):
            left, right = map(float, text.split("-"))
            return (left + right) / 2
        return MISSING_MARKER


def convert_numeric_to_float(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_numeric_to_float)
    return df


def _fix_unit_ml(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "µl": 0.001,
        "μl": 0.001,
        "ul": 0.001,
        "ml": 1,
        "l": 1000,
        "kg": 1000,
        "g": 1,
        "mg": 0.001,
        "cm3": 1,
        "cm³": 1,
        "drop": 0.05,
        "drops": 0.05,
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "ml"
    else:
        return MISSING_MARKER, MISSING_MARKER


def _fix_unit_min(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "s": 1 / 60,
        "sec": 1 / 60,
        "second": 1 / 60,
        "seconds": 1 / 60,
        "min": 1, "mins": 1,
        "minute": 1,
        "minutes": 1,
        "h": 60,
        "hr": 60,
        "hrs": 60,
        "hour": 60,
        "hours": 60,
        "d": 24 * 60,
        "day": 24 * 60,
        "days": 24 * 60,
        "week": 24 * 60 * 7,
        "weeks": 24 * 60 * 7,
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "min"
    else:
        return MISSING_MARKER, MISSING_MARKER


def _fix_unit_c(amount, unit):
    unit = str(unit).strip().lower()
    converters = {
        "°c": lambda x: x,
        "◦c": lambda x: x,
        "c": lambda x: x,
        "k": lambda x: x - 273.15,
    }
    if (unit in converters.keys()) and (amount != MISSING_MARKER):
        return converters[unit](amount), "c"
    else:
        return MISSING_MARKER, MISSING_MARKER


def _fix_unit_hz(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "khz": 1000,
        "hz": 1,
        "hertz": 1,
        "s-1": 1,
        "rpm": 1 / 60
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "hz"
    else:
        return MISSING_MARKER, MISSING_MARKER


def apply_fix_unit(df, unit_cols):
    """
    Applies normalization of units to canonical values
    """
    for target_unit, sets_of_cols in unit_cols.items():
        for cols in sets_of_cols:
            if not all(c in df.columns for c in cols):
                continue
            if target_unit == "ml":
                df[cols] = df.apply(lambda x: _fix_unit_ml(x[cols[0]], x[cols[1]]),
                                     axis="columns", result_type="expand")
            elif target_unit == "c":
                df[cols] = df.apply(lambda x: _fix_unit_c(x[cols[0]], x[cols[1]]),
                                     axis="columns", result_type="expand")
            elif target_unit == "min":
                df[cols] = df.apply(lambda x: _fix_unit_min(x[cols[0]], x[cols[1]]),
                                     axis="columns", result_type="expand")
            elif target_unit == "hz":
                df[cols] = df.apply(lambda x: _fix_unit_hz(x[cols[0]], x[cols[1]]),
                                     axis="columns", result_type="expand")
    return df