from __future__ import annotations
from typing import Optional

import re
import numpy as np
import pandas as pd

from resources.categories import CATEGORIES
from resources.auto_tolerance_bins import AUTO_TOLERANCE_BINS
from resources.config_experiment import NUMERIC_COLUMNS

NAME_OVERRIDES = {
    "Time of Drying": "Drying Time",
    "Temperature of Drying": "Drying Temperature",
}

CATEGORICAL_COLUMN_CHAINS = {
    "Type of Synthesis": ["type_of_synthesis_categories"],
    "Solvent": ["solvent_categories", "solvent_families"],
    "Second Solvent": ["solvent_categories", "solvent_families"],
    "Third Solvent": ["solvent_categories", "solvent_families"],
    "Mixing Apparatus": ["mixing_apparatus_categories"],
}


def map_category(value, mapping: dict):
    """
    Mapping categorical values to CATEGORIES
    """
    if pd.isna(value):
        return value
    
    text = str(value).strip().lower()
    if text in mapping:
        return mapping[text]

    candidates = [key for key in mapping if key.lower() in text]
    if candidates:
        best = max(candidates, key=len)
        return mapping[best]

    return value


def parse_float_value(value) -> Optional[str]:
    """
    Parsing numreical value as numbers or as the average of a range
    """
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except:
        value = re.sub(r'[−–—‐-]', '-', str(value))
        pattern = r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[−–—‐-][+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
        if re.match(pattern, value):
            left, right = map(float, value.split('-'))
            return (left + right) / 2
        return np.nan


def assign_bin(value, bins: list) -> Optional[str]:
    """
    Mapping numerical values to bins from AUTO_TOLERANCE_BINS
    """
    if pd.isna(value):
        return np.nan

    value = parse_float_value(value)

    for lo, hi in bins:
        if hi is None and value >= lo:
            return f"[{lo}, inf)"
        if lo <= value < hi:
            return f"[{lo}, {hi})"

    first_lo, _ = bins[0]
    if value < first_lo:
        lo, hi = bins[0]
        return f"[{lo}, {hi})" if hi is not None else f"[{lo}, inf)"
    
    return np.nan


class Preprocessor:
    """
    Dataset preprocessor
    - transform: discretization of numerical features and normalization of categorical features
    - coverage_report: statistics for processing df_raw
    """
    def __init__(self, method: str = "grinding"):
        self.method = method
        self.bins = AUTO_TOLERANCE_BINS[method]

    def transform(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()

        for column, chain in CATEGORICAL_COLUMN_CHAINS.items():
            if column not in df.columns:
                continue

            series = df[column]
            for dict_name in chain:
                mapping = CATEGORIES[dict_name]
                series = series.apply(lambda v: map_category(v, mapping))
            df[column] = series

        for column in NUMERIC_COLUMNS:
            if column not in df.columns:
                continue
            df[column] = df[column].apply(lambda v: assign_bin(v, self.bins[column]))

        if out_path:
            df.to_csv(out_path, index=False)

        return df

    def coverage_report(self, df_raw: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for column in list(CATEGORICAL_COLUMN_CHAINS) + NUMERIC_COLUMNS:
            if column not in df_raw.columns:
                continue

            observed = df_raw[column].notna()
            n_observed = int(observed.sum())

            if column in CATEGORICAL_COLUMN_CHAINS:
                final_dict = CATEGORICAL_COLUMN_CHAINS[column][-1]
                canonical_values = set(CATEGORIES[final_dict].values())
                unmapped = observed & ~df_model[column].isin(canonical_values)
            else:
                unmapped = observed & df_model[column].isna()
            
            rows.append({
                "column": column,
                "n_observed": n_observed,
                "n_unmapped": int(unmapped.sum()),
            })
        
        return pd.DataFrame(rows)