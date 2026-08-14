from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from resources.categories import CATEGORIES
from resources.auto_tolerance_bins import AUTO_TOLERANCE_BINS
from resources.config_experiment import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from resources.config import MISSING_MARKER, UNIT_COLS

from preprocessing_units import convert_numeric_to_float, apply_fix_unit

CATEGORICAL_COLUMN_CHAINS = {
    "Type of Synthesis": ["type_of_synthesis_categories"],
    "Solvent": ["solvent_categories", "solvent_families"],
    "Second Solvent": ["solvent_categories", "solvent_families"],
    "Third Solvent": ["solvent_categories", "solvent_families"],
    "Mixing Apparatus": ["mixing_apparatus_categories"],
}


def _map_category(value, mapping: dict):
    """
    Mapping categorical values to CATEGORIES
    """
    if value == MISSING_MARKER:
        return value

    text = str(value).strip().lower()
    if text in mapping:
        return mapping[text]

    candidates = [key for key in mapping if key.lower() in text]
    if candidates:
        best = max(candidates, key=len)
        return mapping[best]

    return value


def _map_bin(value, bins: list):
    """
    Mapping numerical values to bins from AUTO_TOLERANCE_BINS
    """
    if value == MISSING_MARKER:
        return MISSING_MARKER

    for lo, hi in bins:
        if hi is None and value >= lo:
            return f"[{lo}, inf)"
        if lo <= value < hi:
            return f"[{lo}, {hi})"
    
    first_lo, _ = bins[0]
    if value < first_lo:
        lo, hi = bins[0]
        return f"[{lo}, {hi})" if hi is not None else f"[{lo}, inf)"

    return MISSING_MARKER


def apply_map_categorical(df: pd.DataFrame, chains=CATEGORICAL_COLUMN_CHAINS, verbose: bool = True):
    """
    Applies mapping to categorical columns in dataset
    """
    for column, chain in chains.items():
        if column not in df.columns:
            continue
        n_before = int((df[column] != MISSING_MARKER).sum())

        series = df[column]
        for dict_name in chain:
            mapping = CATEGORIES[dict_name]
            series = series.apply(lambda v: _map_category(v, mapping))
        df[column] = series

        if verbose:
            canonical_values = set(CATEGORIES[chain[-1]].values())
            n_mapped = int(df[column].isin(canonical_values).sum())
            print(f"[CATEGORICAL] {column}: {n_mapped}/{n_before} mapped to canonical")
    
    return df


def apply_map_bins(df: pd.DataFrame, bins: dict, numeric_columns=NUMERIC_COLUMNS, verbose: bool =True):
    """
    Applies mapping to numeric columns in dataset
    """
    for column in numeric_columns:
        if column not in df.columns:
            continue
        n_before = int((df[column] != MISSING_MARKER).sum())
        df[column] = df[column].apply(lambda v: _map_bin(v, bins[column]))
        if verbose:
            n_after = int((df[column] != MISSING_MARKER).sum())
            print(f"[NUMERIC] {column}: {n_after}/{n_before} binned")
    return df


class Preprocessor:
    """
    Dataset preprocessor
    - transform: discretization and nomalizetion of numerical features, normalization of categorical features
    - coverage_report: statistics for categorical mapping
    - preprocessing_report: statistics for missing values for all preprocessing stages

    """
    def __init__(self, method: str = "grinding"):
        self.method = method
        self.bins = AUTO_TOLERANCE_BINS[method]
        self.unit_cols = UNIT_COLS[method]
        self._stages: dict = {}

    def transform(self, df: pd.DataFrame, out_path: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
        df = df.where(df.notna(), MISSING_MARKER)
        self._stages = {"raw": df.copy()}

        df = convert_numeric_to_float(df, NUMERIC_COLUMNS)
        df = apply_fix_unit(df, self.unit_cols)
        self._stages["after_norm_units"] = df.copy()

        df = apply_map_categorical(df, CATEGORICAL_COLUMN_CHAINS, verbose=verbose)
        self._stages["after_map_cat"] = df.copy()

        df = apply_map_bins(df, self.bins, NUMERIC_COLUMNS, verbose=verbose)
        self._stages["after_map_bins"] = df.copy()

        if out_path:
            df.to_csv(out_path, index=False)

        return df

    def coverage_report(self, df_raw: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.where(df_raw.notna(), MISSING_MARKER)

        rows = []
        for column in list(CATEGORICAL_COLUMN_CHAINS) + NUMERIC_COLUMNS:
            if column not in df_raw.columns:
                continue

            observed = df_raw[column] != MISSING_MARKER
            n_observed = int(observed.sum())

            if column in CATEGORICAL_COLUMN_CHAINS:
                final_dict = CATEGORICAL_COLUMN_CHAINS[column][-1]
                canonical_values = set(CATEGORIES[final_dict].values())
                unmapped = observed & ~df_model[column].isin(canonical_values)
            else:
                unmapped = observed & (df_model[column] == MISSING_MARKER)
            rows.append({"column": column, "n_observed": n_observed, "n_unmapped": int(unmapped.sum())})
        
        return pd.DataFrame(rows)

    def preprocessing_report(self) -> pd.DataFrame:
        stages = [
            ("unit_norm", "raw", "after_norm_units"),
            ("cat_map", "after_norm_units", "after_map_cat"),
            ("num_bin", "after_map_cat", "after_map_bins"),
        ]
 
        columns = list(dict.fromkeys(NUMERIC_COLUMNS + list(CATEGORICAL_COLUMNS)))
 
        first_before_key = stages[0][1]
        last_after_key = stages[-1][2]
 
        rows = []
        for column in columns:
            if column not in self._stages[first_before_key].columns:
                continue
 
            row = {"column": column}
            row["n_observed_start"] = int((self._stages[first_before_key][column] != MISSING_MARKER).sum())
 
            for stage_name, before_key, after_key in stages:
                before, after = self._stages[before_key], self._stages[after_key]
                observed_before = before[column] != MISSING_MARKER
                missing_after = observed_before & (after[column] == MISSING_MARKER)
                row[f"n_missing_{stage_name}"] = int(missing_after.sum())
 
            row["n_observed_final"] = int((self._stages[last_after_key][column] != MISSING_MARKER).sum())
            rows.append(row)
 
        return pd.DataFrame(rows).sort_values(by="n_observed_final", ascending=False)