from __future__ import annotations

import re
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from resourses.schemas.config import MISSING_MARKER

from src.cosyn.normalization.data_cleaning import (
    CASHE_MANUAL,
    cashe_load,
    cashe_save,
    name_to_smiles,
    apply_fix_unit,
    _try_cactus,
    _try_opsin,
    _try_pubchem_cid,
    _try_pubchem_name,
    _try_pubchempy_api,
    _try_chembl,
)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "resourses" / "caches" / "cashe_auto.pickle"

_RESOLVER_EXECUTOR = ThreadPoolExecutor(max_workers=8)


def _with_hard_timeout(fn, hard_timeout_margin=5):
    def wrapped(name, timeout):
        future = _RESOLVER_EXECUTOR.submit(fn, name, timeout)
        try:
            return future.result(timeout=timeout + hard_timeout_margin)
        except FutureTimeoutError:
            return None
        except Exception:
            return None
    return wrapped


RESOLVERS = [
    _with_hard_timeout(fn)
    for fn in (_try_cactus, _try_opsin, _try_pubchem_name, _try_pubchem_cid, _try_pubchempy_api, _try_chembl)
]


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


def prepare_amount_unit_columns(df, shared_col: str = "Amount Unit", targets_col=("Amount Unit of API", "Amount Unit of Coformer")):
    """
    Splits the shared unit column into per-compound unit columns
    """
    if shared_col in df.columns:
        for target in targets_col:
            if target not in df.columns:
                df[target] = df[shared_col]
    return df


def resolve_amount_smiles(df, name_cols=("API", "Coformer"), cache_path: Path = DEFAULT_CACHE_PATH,
                          timeout: int = 8, use_remote: bool = True, verbose: bool = True):
    """
    Resolves compound names in name_cols to SMILES, needed to compute molecular weight. Results are cached to disk
    """
    cache = cashe_load(cache_path)
    cache.update(CASHE_MANUAL)

    invalid_names = set()
    unique_names = set()

    for col in name_cols:
        if col in df.columns:
            unique_names.update(df[col].dropna().unique())
    unique_names.discard(MISSING_MARKER)

    resolved = {MISSING_MARKER: MISSING_MARKER}
    iterator = tqdm(sorted(unique_names), desc="Resolving SMILES for amount normalization") if verbose else sorted(unique_names)
    
    for i, name in enumerate(iterator):
        resolved[name] = name_to_smiles(name, cache, timeout, RESOLVERS, invalid_names, use_remote=use_remote) or MISSING_MARKER
        if use_remote and (i + 1) % 10 == 0:
            cashe_save(cache, cache_path)

    if use_remote:
        cashe_save(cache, cache_path)

    for col in name_cols:
        if col in df.columns:
            df[f"{col} SMILES"] = df[col].map(resolved)

    if verbose:
        print(f"[SMILES] resolved {sum(v != MISSING_MARKER for v in resolved.values())}/{len(resolved)} unique names")

    return df


def amount_unit_cols_with_smiles(unit_cols, name_cols=("API", "Coformer"), smiles_suffix: str = " SMILES"):
    """
    Returns a copy of unit_cols where the "mg" entries point at the resolved "<name_col> SMILES" columns instead of the raw name columns
    """
    updated = {target_unit: [list(cols) for cols in sets_of_cols] for target_unit, sets_of_cols in unit_cols.items()}
    for cols in updated.get("mg", []):
        if cols and cols[0] in name_cols:
            cols[0] = f"{cols[0]}{smiles_suffix}"
    return updated


def apply_unit_normalization(df, unit_cols, name_cols=("API", "Coformer"), cache_path: Path = DEFAULT_CACHE_PATH, use_remote: bool = True, verbose: bool = True):
    """
    Applies unit normalization for all target units in unit_cols
    """
    df = prepare_amount_unit_columns(df)
    df = resolve_amount_smiles(df, name_cols=name_cols, cache_path=cache_path, use_remote=use_remote, verbose=verbose)

    unit_cols = amount_unit_cols_with_smiles(unit_cols, name_cols=name_cols)
    df = apply_fix_unit(df, unit_cols)
    return df