import argparse
import pickle
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
import unicodedata

import pandas as pd
import pubchempy as pcp
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm

from resourses.caches.cashe_manual import CASHE_MANUAL
from resourses.mappings.categories import CATEGORIES
from resourses.schemas.config import *

RDLogger.DisableLog("rdApp.*")

ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


# --- Cache helpers ---

DEFAULT_INPUT_DIR = Path("data/extraction/outputs/clean")
DEFAULT_RAW_OUTPUT_DIR = Path("data/normalization/raw")
DEFAULT_OUTPUT_DIR = Path("data/normalization/clean")
DEFAULT_CACHE_PATH = Path("resourses/caches/cashe_auto.pickle")


def cashe_save(cashe, cache_path: Path = DEFAULT_CACHE_PATH):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cashe, f, pickle.HIGHEST_PROTOCOL)


def cashe_load(cache_path: Path = DEFAULT_CACHE_PATH):
    if not cache_path.exists():
        return {}
    with cache_path.open("rb") as f:
        return pickle.load(f)

# --- Name to SMILES ---

def _try_opsin(name, timeout):
    try:
        url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(name, safe='')}.json"
        r = requests.get(
            url, timeout=timeout, headers={"User-Agent": "name2smiles/1.0"}
        )
        if r.ok:
            return (r.json() or {}).get("smiles")
    except Exception:
        return None


def _try_cactus(name, timeout):
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(name, safe='')}/smiles"
        r = requests.get(
            url, timeout=timeout, headers={"User-Agent": "name2smiles/1.0"}
        )
        if r.ok and r.text.strip():
            return r.text.strip()
    except Exception:
        return None


def _try_pubchempy_api(name, timeout):
    try:
        results = pcp.get_compounds(name, "name")
        if results:
            c = results[0]
            return (
                getattr(c, "canonical_smiles", None)
                or getattr(c, "isomeric_smiles", None)
                or getattr(c, "smiles", None)
            )
    except Exception:
        return None


def _try_pubchem_name(name, timeout):
    enc = quote(name.strip(), safe="")
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    try:
        r = requests.get(
            f"{base}/name/{enc}/property/CanonicalSMILES/TXT", timeout=timeout
        )
        t = r.text.strip()
        if r.ok and t and not t.startswith("Status:"):
            return t
    except Exception:
        return None


def _try_pubchem_cid(name, timeout):
    enc = quote(name.strip(), safe="")
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    try:
        r = requests.get(f"{base}/name/{enc}/cids/TXT", timeout=timeout)
        cid = r.text.strip().splitlines()[0]
        r = requests.get(
            f"{base}/cid/{cid}/property/CanonicalSMILES/TXT", timeout=timeout
        )
        t = r.text.strip()
        if r.ok and t and not t.startswith("Status:"):
            return t
    except Exception:
        return None


def _try_chembl(name, timeout):
    try:
        enc = quote(name, safe="")
        url1 = (
            f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?pref_name__iexact={enc}"
        )
        r = requests.get(
            url1, headers={"User-Agent": "name2smiles/1.0"}, timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        molecules = data.get("molecules")
        if not molecules:
            url2 = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={enc}"
            r = requests.get(
                url2, headers={"User-Agent": "name2smiles/1.0"}, timeout=timeout
            )
            r.raise_for_status()
            data = r.json()
            molecules = data.get("molecules")
        structures = molecules[0].get("molecule_structures")
        return structures.get("canonical_smiles")
    except Exception:
        return None


def name_to_smiles(name, cache, timeout, resolvers, invalid_names, use_remote: bool = True):
    if name in cache:
        mol = Chem.MolFromSmiles(cache[name])
        return Chem.MolToSmiles(mol) if mol else None
    if not use_remote:
        invalid_names.add(name)
        return None
    for resolver in resolvers:
        try:
            smiles = resolver(name, timeout)
            if smiles:
                cache[name] = smiles
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            continue
    invalid_names.add(name)
    return None


def convert_names_to_smiles(df, smiles_cols, resolvers, cache_only: bool = False, cache_path: Path = DEFAULT_CACHE_PATH):
    cashe_auto = cashe_load(cache_path)
    cashe_auto.update(CASHE_MANUAL)
    cashe = cashe_auto.copy()
    for col in smiles_cols:
        invalid_names = set()
        tqdm.pandas(desc=f"Searching SMILES for names of {col}")
        df[col] = df[col].progress_apply(
            lambda x: name_to_smiles(
                x, cashe, 10, resolvers, invalid_names,
                use_remote=not cache_only
            )
            if x != MISSING_MARKER
            else None
        )
        if cache_only:
            print(
                f"[CACHE-ONLY] Not found in cache for {col}: {len(invalid_names)} names:",
                invalid_names,
            )
        else:
            print(
                f"Add SMILES to cashe_manual for following {len(invalid_names)} names:",
                invalid_names,
            )
        cashe_save(cashe, cache_path)
    return df


def remove_missing_smiles(df, smiles_cols):
    return df.dropna(subset=smiles_cols).reset_index(drop=True)


def print_rows(df, step):
    print(f"[SMILES] {step}: {len(df)}")


def keep(df, mask, step):
    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=df.index)
    before = len(df)
    df = df.loc[mask]
    after = len(df)
    print(f"[SMILES] {step}: {after}/{before} (-{before-after})")
    return df


def mol_stats(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    fragments = len(Chem.GetMolFrags(mol))
    heavy_atoms = mol.GetNumHeavyAtoms()
    carbon_atoms = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    formal_charge = Chem.GetFormalCharge(mol)
    allowed_elements = all(a.GetAtomicNum() in ALLOWED_ATOMIC_NUMS for a in mol.GetAtoms())
    return canonical, fragments, heavy_atoms, carbon_atoms, formal_charge, allowed_elements

def filter_cocrystal_pairs(
    df,
    api_col="API",
    coformer_col="Coformer",
    min_heavy_api=3,
    min_heavy_coformer=3,
    require_single_fragment=True,
    require_neutral=True,
    require_allowed_elements=True,
):
    print_rows(df, "start")

    api_stats = df[api_col].map(mol_stats)
    cof_stats = df[coformer_col].map(mol_stats)

    df = keep(df, api_stats.notna() & cof_stats.notna(), "rdkit_parseable")
    api_stats, cof_stats = api_stats.loc[df.index], cof_stats.loc[df.index]

    api_canonical = api_stats.map(lambda t: t[0])
    api_fragments = api_stats.map(lambda t: t[1])
    api_heavy = api_stats.map(lambda t: t[2])
    api_carbons = api_stats.map(lambda t: t[3])
    api_charge = api_stats.map(lambda t: t[4])
    api_allowed = api_stats.map(lambda t: t[5])

    cof_canonical = cof_stats.map(lambda t: t[0])
    cof_fragments = cof_stats.map(lambda t: t[1])
    cof_heavy = cof_stats.map(lambda t: t[2])
    cof_carbons = cof_stats.map(lambda t: t[3])
    cof_charge = cof_stats.map(lambda t: t[4])
    cof_allowed = cof_stats.map(lambda t: t[5])

    df = keep(df, api_canonical.ne(cof_canonical), "api_not_coformer")
    api_fragments, cof_fragments = api_fragments.loc[df.index], cof_fragments.loc[df.index]
    api_heavy, cof_heavy = api_heavy.loc[df.index], cof_heavy.loc[df.index]
    api_carbons, cof_carbons = api_carbons.loc[df.index], cof_carbons.loc[df.index]
    api_charge, cof_charge = api_charge.loc[df.index], cof_charge.loc[df.index]
    api_allowed, cof_allowed = api_allowed.loc[df.index], cof_allowed.loc[df.index]

    if require_single_fragment:
        df = keep(df, (api_fragments == 1) & (cof_fragments == 1), "single_fragment")
        api_heavy, cof_heavy = api_heavy.loc[df.index], cof_heavy.loc[df.index]
        api_carbons, cof_carbons = api_carbons.loc[df.index], cof_carbons.loc[df.index]
        api_charge, cof_charge = api_charge.loc[df.index], cof_charge.loc[df.index]
        api_allowed, cof_allowed = api_allowed.loc[df.index], cof_allowed.loc[df.index]

    df = keep(df, (api_carbons >= 1) & (cof_carbons >= 1), "has_carbon")
    api_heavy, cof_heavy = api_heavy.loc[df.index], cof_heavy.loc[df.index]
    api_charge, cof_charge = api_charge.loc[df.index], cof_charge.loc[df.index]
    api_allowed, cof_allowed = api_allowed.loc[df.index], cof_allowed.loc[df.index]

    df = keep(df, (api_heavy >= min_heavy_api) & (cof_heavy >= min_heavy_coformer), "min_heavy_atoms")
    api_charge, cof_charge = api_charge.loc[df.index], cof_charge.loc[df.index]
    api_allowed, cof_allowed = api_allowed.loc[df.index], cof_allowed.loc[df.index]

    if require_neutral:
        df = keep(df, (api_charge == 0) & (cof_charge == 0), "neutral")
        api_allowed, cof_allowed = api_allowed.loc[df.index], cof_allowed.loc[df.index]

    if require_allowed_elements:
        df = keep(df, api_allowed & cof_allowed, "allowed_elements")

    df = df.reset_index(drop=True)
    print_rows(df, "end")
    return df


# --- Cleaning helpers ---

def delete_rows_wo_info(df, NECESSARY_COLS):
    for col in NECESSARY_COLS:
        if col in df.columns:
            df = df[~(df[col] == MISSING_MARKER)]
    return df


def numeric_to_float(num):
    try:
        return float(num)
    except Exception:
        return MISSING_MARKER


def convert_numeric_to_float(df, method, numeric_cols):
    for col in numeric_cols.get(method, []):
        df[col] = df[col].apply(numeric_to_float)
    return df


def convert_string_to_str(df, method, string_cols):
    for col in string_cols.get(method, []):
        df[col] = df[col].astype(str)
    return df


def clear_units_by_dependency(df, unit_depencencies):
    for unit_col, amount_candidates in unit_depencencies.items():
        if unit_col not in df.columns:
            continue
        amount_col = next((c for c in amount_candidates if c in df.columns), None)
        if amount_col is None:
            continue
        mask = df[amount_col].eq(MISSING_MARKER)
        df.loc[mask, unit_col] = MISSING_MARKER
    return df


def edit_columns(df, replace_cols, remove_unit_cols, unit_depencencies):
    for col_new, col_old in replace_cols.items():
        if col_old not in df.columns:
            continue
        df[col_new] = df[col_old]
        df = clear_units_by_dependency(df, unit_depencencies)
    return df.drop(columns=remove_unit_cols, errors="ignore")


def _to_numeric(series, missing_marker):
    return pd.to_numeric(series.mask(series.eq(missing_marker)), errors="coerce")

def apply_domain_rules(df, method, parameters_ranges, MISSING_MARKER):
    rules_by_col = parameters_ranges.get(method, {})
    for col, rule in rules_by_col.items():
        if col not in df.columns:
            continue
        raw = df[col]
        present = raw.notna() & raw.ne(MISSING_MARKER)
        if not present.any():
            continue
        num = _to_numeric(raw, MISSING_MARKER)
        min_v, max_v = rule.get("min", None), rule.get("max", None)
        bad = present & (num.isna()
                         | ((min_v is not None) & num.lt(min_v))
                         | ((max_v is not None) & num.gt(max_v)))
        if bad.any():
            df.loc[bad, col] = MISSING_MARKER
    return df

def apply_percentile_tail_cut(df, method, PARAMETERS_RANGES, MISSING_MARKER, lower_q=0.001, upper_q=0.999, min_n=200):
    cols = PARAMETERS_RANGES.get(method, {}).keys()
    for col in cols:
        if col not in df.columns:
            continue
        num = _to_numeric(df[col], MISSING_MARKER)
        ok = num.notna()
        if int(ok.sum()) < min_n:
            continue
        low, high = float(num[ok].quantile(lower_q)), float(num[ok].quantile(upper_q))
        out = ok & (num.lt(low) | num.gt(high))
        if out.any():
            df.loc[out, col] = MISSING_MARKER
    return df


def _fix_synthesis_type(synthesis_type, additional_value, method, type_of_synthesis_categories):
    synthesis_type_l = str(synthesis_type).lower()
    for key, value in type_of_synthesis_categories.items():
        if key in synthesis_type_l:
            synthesis_type = value
    if method == "grinding":
        if (synthesis_type_l == "") and (additional_value != MISSING_MARKER):
            synthesis_type = "liquid-assisted grinding"
        if (synthesis_type_l == "") and (additional_value == MISSING_MARKER):
            synthesis_type = "neat grinding"
        if (synthesis_type_l == "neat grinding") and (additional_value != MISSING_MARKER):
            synthesis_type = "liquid-assisted grinding"
    return synthesis_type


def apply_fix_synthesis_type(df, method, type_of_synthesis_categories):
    if "Type of Synthesis" in df.columns:
        if method in ["grinding", "solution_crystallization"]:
            df["Type of Synthesis"] = df.apply(
                lambda x: _fix_synthesis_type(x["Type of Synthesis"], x["Solvent"], 
                                              method, type_of_synthesis_categories), axis=1)
        else:
            df["Type of Synthesis"] = df.apply(
                lambda x: _fix_synthesis_type(x["Type of Synthesis"], None, 
                                              method, type_of_synthesis_categories), axis=1)
    return df


def _fix_ratio(ratio_api, ratio_cof):
    if ((ratio_api == MISSING_MARKER) | (ratio_cof == MISSING_MARKER) 
        | (ratio_api == ratio_cof) | (ratio_api == 0) | (ratio_cof == 0)):
        ratio_api, ratio_cof = 1, 1
    else:
        f = (Fraction(ratio_api).limit_denominator() / Fraction(ratio_cof).limit_denominator())
        ratio_api, ratio_cof = f.numerator, f.denominator
        if (ratio_api > 10) | (ratio_cof > 10):
            ratio_api, ratio_cof = 1, 1
    return int(ratio_api), int(ratio_cof)


def apply_fix_ratio(df, RATIO_COLS):
    df[RATIO_COLS] = df.apply(lambda x: _fix_ratio(x[RATIO_COLS[0]], x[RATIO_COLS[1]]), axis="columns", result_type="expand")
    return df


def _normalize_solvent(solvent) :
    if solvent is None:
        return MISSING_MARKER
    s = str(solvent)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("µ", "u")
    s = s.replace("®", "")
    s = s.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    s = s.replace("γ", "gamma").replace("λ", "lambda").replace("ε", "epsilon")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def apply_fix_solvent(df, solvent_categories, solvent_cols):
    for col in solvent_cols:
        df[col] = (
            df[col]
            .map(_normalize_solvent)
            .map(solvent_categories)
            .fillna(MISSING_MARKER)
        )
    return df

def _fix_mixing_apparatus(apparatus, amount, unit, apparatus_categories):
    patterns = [
        (r"retsch.*mm200", "ball mill"),
        (r"retsch.*mm400", "ball mill"),
        (r"retsch.*mm300", "ball mill"),
        (r"retsch.*mm301", "ball mill"),
        (r"spex", "vibration mill"),
        (r"fritsch.*pulverisette", "planetary mill"),
        (r"oscillatory.*mill", "vibration mill"),
        (r"vibrat.*mill", "vibration mill"),
    ]
    apparatus_l = str(apparatus).lower()
    for pattern, category in patterns:
        if re.search(pattern, apparatus_l):
            return category, amount, unit

    for key, value in sorted(apparatus_categories.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in apparatus_l:
            if value == "mortar and pestle":
                amount, unit = MISSING_MARKER, MISSING_MARKER
            return value, amount, unit

    return MISSING_MARKER, MISSING_MARKER, MISSING_MARKER


def _fix_ultrasonic_apparatus(apparatus, apparatus_categories):
    apparatus_l = str(apparatus).lower()
    for key, value in apparatus_categories.items():
        if key in apparatus_l:
            return value
    return MISSING_MARKER
        

def apply_fix_apparatus(df, method, apparatus_categories, apparatus_cols):
    if not apparatus_cols:
        return df
    if method == "grinding":
        df[apparatus_cols] = df.apply(
            lambda x: _fix_mixing_apparatus(
                x[apparatus_cols[0]],
                x[apparatus_cols[1]],
                x[apparatus_cols[2]],
                apparatus_categories,
            ),
            axis="columns",
            result_type="expand",
        )
    if method == "ultrasound":
        df[apparatus_cols[0]] = df[apparatus_cols[0]].apply(
            lambda x: _fix_ultrasonic_apparatus(x, apparatus_categories
            )  
        )
    return df


def _fix_covering_method(covering, covering_categories):
    covering = str(covering).lower()
    if covering in covering_categories:
        return covering_categories[covering]
    return MISSING_MARKER


def _fix_description_of_holes(holes, holes_categories):
    holes = str(holes).lower()
    if holes in holes_categories:
        return holes_categories[holes]
    return MISSING_MARKER


def apply_fix_covering_and_holes(df, covering_categories, covering_col, holes_categories, holes_col):
    if not covering_col or covering_col not in df.columns:
        return df
    def _row_fix(row):
        covering_std = _fix_covering_method(row[covering_col], covering_categories)
        if covering_std in ("open setup", "cap"):
            holes_std = "no holes"
        else:
            holes_std = _fix_description_of_holes(row[holes_col], holes_categories)
        return covering_std, holes_std
    df[[covering_col, holes_col]] = df.apply(_row_fix, axis=1, result_type="expand")
    return df


# --- Unit conversions ---

def _fix_unit_ml(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "µl": 0.001,
        "μl": 0.001,
        "ul": 0.001,
        "ml": 1,
        "drops from a Pasteur pipette":  0.05,
        "drops": 0.05,
        "drop": 0.05,
        "ll": 1,
        ":l": 1,
        "kg": 1000,
        "g": 1,        # assumption 1 g ≈ 1 mL if density ~1
        "mg": 0.001,   # 1 mg ≈ 0.001 mL
        "l": 1000,     # 1 L = 1000 mL
        "cm3": 1,
        "cm³": 1,
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "ml"
    else:
        return MISSING_MARKER, MISSING_MARKER


def _fix_unit_mg(smiles, amount, unit):
    unit = str(unit).strip().lower()
    mol = Chem.MolFromSmiles(smiles) if smiles not in (None, MISSING_MARKER) else None
    if mol is None:
        return smiles, MISSING_MARKER, MISSING_MARKER
    mw = Descriptors.MolWt(mol)
    conversion_factors = {
        "mg": 1,              # mg
        "milligrams": 1,
        "g": 1000,            # 1 g = 1000 mg
        "kg": 1000 * 1000,            
        "m": mw * 1000,
        "mol": mw * 1000,     # mol → mg
        "mole": mw * 1000,
        "moles": mw * 1000,
        "mmol": mw,
        "mmole": mw,
        "μmol": mw / 1000,     
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return smiles, amount * conversion_factors[unit], "mg"
    else:
        return smiles, MISSING_MARKER, MISSING_MARKER


def _fix_unit_min(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "s": 1 / 60,
        "sec": 1 / 60,
        "second": 1 / 60,
        "seconds": 1 / 60,
        "min": 1,
        "mins": 1,
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
        "month": 24 * 60 * 7 * 30,
        "months": 24 * 60 * 7 * 30,
        "year":  24 * 60 * 7 * 30 * 365,
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
        "c":  lambda x: x,
        "k":  lambda x: x - 273.15,
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
        "rpm": 1 / 60}
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "hz"
    else:
        return MISSING_MARKER, MISSING_MARKER
    

def _fix_unit_ml_min(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "ml/h": 1/60,
        "ml/min": 1
    }
    if (unit in conversion_factors.keys()) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "ml/min"
    else:
        return MISSING_MARKER, MISSING_MARKER
    

def _fix_unit_mm(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "mm": 1, 
        "millimeter": 1, 
        "cm": 10, "m": 1000, 
        "um": 0.001, 
        "µm": 0.001, 
        "in": 25.4, 
        "inch": 25.4
    }
    if (unit in conversion_factors) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "mm"
    return MISSING_MARKER, MISSING_MARKER
    

def _fix_unit_w(amount, unit):
    unit = str(unit).strip().lower()
    conversion_factors = {
        "w": 1, 
        "mw": 0.001, 
        "kw": 1000}
    if (unit in conversion_factors) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "w"
    return MISSING_MARKER, MISSING_MARKER


def _fix_unit_mm_s(amount, unit):
    unit = str(unit).strip().lower().replace(" ", "")
    conversion_factors = {
        "mm/s": 1,
        "mms-1": 1,
        "cm/s": 10,
        "cms-1": 10,
        "m/s": 1000,
        "ms-1": 1000,
        "mm/min": 1 / 60,
        "cm/min": 10 / 60,
    }
    if (unit in conversion_factors) and (amount != MISSING_MARKER):
        return amount * conversion_factors[unit], "mm/s"
    return MISSING_MARKER, MISSING_MARKER


def apply_fix_unit(df, unit_cols):
    for target_unit, sets_of_cols in unit_cols.items():
        for cols in sets_of_cols:
            if not all(c in df.columns for c in cols):
                continue
            if target_unit == "mg":
                df[cols] = df.apply(lambda x: _fix_unit_mg(x[cols[0]], x[cols[1]], x[cols[2]]),
                                   axis="columns", result_type="expand")
            elif target_unit == "ml":
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
            elif target_unit == "ml/min":
                df[cols] = df.apply(lambda x: _fix_unit_ml_min(x[cols[0]], x[cols[1]]),
                                   axis="columns", result_type="expand")
            elif target_unit == "mm":
                df[cols] = df.apply(lambda x: _fix_unit_mm(x[cols[0]], x[cols[1]]),
                                   axis="columns", result_type="expand")
            elif target_unit == "w":
                df[cols] = df.apply(lambda x: _fix_unit_w(x[cols[0]], x[cols[1]]),
                                   axis="columns", result_type="expand")
            elif target_unit == "mm/s":
                df[cols] = df.apply(lambda x: _fix_unit_mm_s(x[cols[0]], x[cols[1]]),
                                   axis="columns", result_type="expand")
    return df


METHOD_CHOICES = [
    "grinding",
    "slurry",
    "solution_crystallization",
    "antisolvent",
    "laser_irradiation",
    "ultrasound",
    "freeze_drying",
    "hot_melt_extrusion",
    "resonant_acoustic",
    "spray_drying",
]


# --- Main driver ---

def process_method(
    method: str,
    input_dir: Path = DEFAULT_INPUT_DIR,
    raw_output_dir: Path = DEFAULT_RAW_OUTPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_path: Path = DEFAULT_CACHE_PATH,
    cache_only: bool = False,
):

    RESOLVERS = [_try_cactus, _try_opsin, _try_pubchem_name, _try_pubchem_cid, _try_pubchempy_api, _try_chembl]

    file_path = input_dir / f"{method}.csv"
    raw_file_path = raw_output_dir / f"{method}.csv"
    output_file_path = output_dir / f"{method}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    raw_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)
    stats = {
        "method": method,
        "input_path": str(file_path),
        "raw_output_path": str(raw_file_path),
        "clean_output_path": str(output_file_path),
        "cache_only": bool(cache_only),
        "status": "started",
        "rows_input": int(len(df)),
        "cols_input": int(df.shape[1]),
    }

    # Method-specific unit column handling
    df = edit_columns(df, REPLACE_COLS.get(method, {}), REMOVE_UNIT_COLS.get(method, []), UNIT_DEPENDENCIES)
    stats["rows_after_column_edit"] = int(len(df))
    stats["cols_after_column_edit"] = int(df.shape[1])

    # If dataset have small amount of rows save just raw variant
    if method in METHODS_SMALL:
        df[FINAL_COLS[method]].to_csv(raw_file_path, index=False)
        stats.update({
            "status": "raw_only_small_dataset",
            "rows_raw_saved": int(len(df)),
            "rows_output": int(len(df)),
            "rows_removed_total": 0,
        })
        print(f"[CLEAN] {method}: small dataset saved to raw only: {raw_file_path}")
        return stats

    # Export datasets with fixed columns
    df.to_csv(raw_file_path, index=False)
    stats["rows_raw_saved"] = int(len(df))

    # Remove rows without API/Coformer
    df = delete_rows_wo_info(df, NECESSARY_COLS)
    stats["rows_after_required_info"] = int(len(df))

    # Convert numerics
    df = convert_numeric_to_float(df, method, NUMERIC_COLS)

    # Convert string
    df = convert_string_to_str(df, method, STRING_COLS)

    # Names to SMILES (overwrites API / Coformer with SMILES)
    df = convert_names_to_smiles(df, SMILES_COLS, RESOLVERS, cache_only=cache_only, cache_path=cache_path)
    stats["rows_after_name_to_smiles"] = int(len(df))
    df = remove_missing_smiles(df, SMILES_COLS)
    stats["rows_after_remove_missing_smiles"] = int(len(df))
    df = filter_cocrystal_pairs(df, api_col="API", coformer_col="Coformer", min_heavy_api=3, min_heavy_coformer=3, 
                                require_single_fragment=True, require_neutral=True, require_allowed_elements=True)
    stats["rows_after_cocrystal_filter"] = int(len(df))

    # Synthesis type (only affects methods that actually have this column)
    df = apply_fix_synthesis_type(df, method, CATEGORIES["type_of_synthesis_categories"])

    # Ratio
    df = apply_fix_ratio(df, RATIO_COLS)

    # Solvent normalisation
    df = apply_fix_solvent(df, CATEGORIES["solvent_categories"], SOLVENT_COLS[method])

    # Mixing pparatus
    df = apply_fix_apparatus(df, method, CATEGORIES["mixing_apparatus_categories"], APPARATUS_COLS[method])

    # Covering and Holes
    cols = COVERING_AND_HOLES_COLS.get(method, [])
    covering_col = cols[0] if len(cols) > 0 else None
    holes_col = cols[1] if len(cols) > 1 else None    
    df = apply_fix_covering_and_holes(
        df,
        CATEGORIES["covering_method_categories"],
        covering_col,
        CATEGORIES["description_of_holes_categories"],
        holes_col,
    )

    # Unit conversions
    df = apply_fix_unit(df, UNIT_COLS[method])

    # Handle with outliers
    df = apply_domain_rules(df, method, PARAMETERS_RANGES, MISSING_MARKER)
    df = apply_percentile_tail_cut(df, method, PARAMETERS_RANGES, MISSING_MARKER, lower_q=0.02, upper_q=0.98, min_n=100)

    # Export final columns
    df[FINAL_COLS[method]].to_csv(output_file_path, index=False)
    stats.update({
        "status": "cleaned",
        "rows_output": int(len(df)),
        "cols_output": int(len(FINAL_COLS[method])),
        "rows_removed_total": int(stats["rows_input"] - len(df)),
    })
    print(f"[CLEAN] {method}: rows={len(df)} saved to {output_file_path}")
    return stats


def main(
    method: str | None = None,
    methods: list[str] | None = None,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    raw_output_dir: str | Path = DEFAULT_RAW_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    cache_only: bool = False,
):
    input_dir = Path(input_dir)
    raw_output_dir = Path(raw_output_dir)
    output_dir = Path(output_dir)
    cache_path = Path(cache_path)

    selected_methods = methods or ([method] if method else list(METHODS) + list(METHODS_SMALL))

    stats = []
    for selected_method in selected_methods:
        stats.append(process_method(
            selected_method,
            input_dir=input_dir,
            raw_output_dir=raw_output_dir,
            output_dir=output_dir,
            cache_path=cache_path,
            cache_only=cache_only,
        ))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--method",
        type=str,
        choices=METHOD_CHOICES,
        help="Process one method CSV.",
    )
    group.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        help="Process selected method CSVs.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all supported method CSVs.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    selected = METHOD_CHOICES if args.all else args.methods
    main(
        method=args.method,
        methods=selected,
        input_dir=args.input_dir,
        raw_output_dir=args.raw_output_dir,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        cache_only=args.cache_only,
    )
