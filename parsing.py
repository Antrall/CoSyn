from typing import Optional
import pandas as pd

from resources.schemas.config_experiment import UNIT_COLUMNS

NAME_OVERRIDES = {
    "Time of Drying": "Drying Time",
    "Temperature of Drying": "Drying Temperature",
}

UNIT_NORMALIZATION = {
    "minute": "min",
    "minutes": "min",
    "mins": "min",
    "h": "hrs",
    "hour": "hrs",
    "hours": "hrs",
    "hrs": "hrs",
    "seconds": "sec",
    "s": "sec",

    "◦c": "c",
    "° c": "c",
    "°C": "c",
    "◦C": 'c',
    "C": "c",
    "degc": "c",

    "Hz": "hz",
    "Hertz": "hz",
    "HZ": "hz",
    "beats per second": "bps",
    "beats/s": "bps",

    "μL": "μl",
    "µL": "μl",
    "µl": "μl",
    "uL": "μl",
    "ul": "μl",
    "mL": "ml",

    "moles": "mol"
}


def parse_raw_csv(path: str, verbose: bool = True, out_path: Optional[str] = None) -> pd.DataFrame:
    """
    Parsing and converting raw CSV data into a DataFrame.
    - Each row — each piece of information about the synthesis
    - Units from UNIT_COLUMNS are normalized like UNIT_NORMALIZATION
    """
    df = pd.read_csv(path).drop('answer', axis=1)
    df = df.rename(columns=NAME_OVERRIDES)

    for col in UNIT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: UNIT_NORMALIZATION.get(str(v), v)
                if pd.notna(v)
                else v
            )

    if verbose:
        print(f"Articles detected:  {len(df)}")
        print(f"Unique article ids: {len(df['ids'].unique())}")

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Saved cleaned dataset to: {out_path}")

    return df