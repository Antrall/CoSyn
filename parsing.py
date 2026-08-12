from typing import Optional
import pandas as pd

MISSING_SENTINEL = "NOT_DETECTED"

UNIT_COLUMNS = [
    "Amount Unit",
    "Solvent Amount Unit",
    "Frequency Unit",
    "Mixing Time Unit",
    "Mixing Temperature Unit",
    "Drying Time Unit",
    "Drying Temperature Unit",
]

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
    Парсинг и преобразование необработанного CSV в DataFrame
    - Каждая строка соответствует каждой записи о синтезе
    - При наличии MISSING_SENTINEL в ячейке он заменяется на NaN
    - Единицы из столбцов UNIT_COLUMNS нормализуются в соответствии с UNIT_NORMALIZATION
    """
    df = pd.read_csv(path).drop('answer', axis=1)
    df = df.replace(MISSING_SENTINEL, pd.NA)

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