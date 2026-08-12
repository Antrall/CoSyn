from __future__ import annotations

import pandas as pd

CANONICAL_UNITS = {
    "Amount Unit": "mg",
    "Solvent Amount Unit": "ml",
    "Second Solvent Amount Unit": "ml",
    "Third Solvent Amount Unit": "ml",
    "Frequency Unit": "hz",
    "Mixing Temperature Unit": "c",
    "Drying Temperature Unit": "c",
    "Mixing Time Unit": "min",
    "Drying Time Unit": "min",
}

UNIT_DIMENSION_GROUPS = {
    "mass": {"mg", "g", "kg", "µg", "ug"},
    "amount": {"mol", "mmol", "μmol", "umol"},
    "volume": {"ml", "l", "μl", "ul", "cm3", "cc"},
    "time": {"min", "h", "hrs", "hour", "hours", "s", "sec", "seconds", "days"},
    "temperature": {"c", "k", "f"},
    "frequency": {"hz", "rpm"},
}


def _dimension_of(unit: str) -> str | None:
    unit = unit.strip().lower()
    for dimension, units in UNIT_DIMENSION_GROUPS.items():
        if unit in units:
            return dimension
    return None


def classify_unit(raw_unit: str, canonical_unit: str) -> str:
    """
    Классифицирует единицу измерения относительно эталонной:
    - canonical: совпадает с эталонной из CANONICAL_UNITS
    - same dimension: конвертируется через простые операции
    - another dimension: требует внешних данных для конвертации
    - unknown: указана, но не распознана словарём
    - no data: отсутствие данных
    """
    if pd.isna(raw_unit):
        return "no data"

    raw_norm = str(raw_unit).strip().lower()
    canon_norm = canonical_unit.strip().lower()

    if raw_norm == canon_norm:
        return "canonical"

    raw_dim = _dimension_of(raw_norm)
    canon_dim = _dimension_of(canon_norm)

    if raw_dim is None or canon_dim is None:
        return "unknown"
    return "same dimension" if raw_dim == canon_dim else "another dimension"


def unit_dimension_report(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Сводка по Unit-колонкам датасета
    """
    unit_columns = tuple(CANONICAL_UNITS)
    canonical_units = CANONICAL_UNITS

    rows = []
    for column in unit_columns:
        if column not in df.columns:
            continue
    
        canon = canonical_units[column]
        counts = df[column].value_counts(dropna=False)

        for raw_value, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "canonical_unit": canon,
                    "raw_value": raw_value,
                    "count": int(count),
                    "category": classify_unit(raw_value, canon),
                }
            )
    return pd.DataFrame(rows).sort_values(["column", "count"], ascending=[True, False])