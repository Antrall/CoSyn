###  All columns

RAW_COLUMNS = [
    'ids',
    'doi',
    'Type of Synthesis',
    'Formula',
    'API',
    'Coformer',
    'Part of Ratio API',
    'Part of Ratio Coformer',
    'Amount of API',
    'Amount of Coformer',
    'Amount Unit',
    'Solvent',
    'Amount of Solvent',
    'Solvent Amount Unit',
    'Second Solvent',
    'Amount of Second Solvent',
    'Third Solvent',
    'Amount of Third Solvent',
    'Mixing Apparatus',
    'Mixing Frequency',
    'Frequency Unit',
    'Mixing Time',
    'Mixing Time Unit',
    'Mixing Temperature',
    'Mixing Temperature Unit',
    'Time of Drying',
    'Drying Time Unit',
    'Temperature of Drying',
    'Drying Temperature Unit',
    'Temperature of Drying (numeric)'
]

COLUMNS = [
    'ids',
    'doi',
    'Type of Synthesis',
    'Formula',
    # ----
    'API',
    'Coformer',
    'API SMILES',
    'Coformer SMILES',
    'Part of Ratio API',
    'Part of Ratio Coformer',
    'Amount of API',
    'Amount of Coformer',
    'Amount Unit of API',
    'Amount Unit of Coformer',
    # ----
    'Solvent',
    'Amount of Solvent',
    'Second Solvent',
    'Amount of Second Solvent',
    'Third Solvent',
    'Amount of Third Solvent',
    'Solvent Amount Unit',
    # ----
    'Mixing Apparatus',
    'Mixing Frequency',
    'Frequency Unit',
    'Mixing Time',
    'Mixing Time Unit',
    'Mixing Temperature',
    'Mixing Temperature Unit',
    # ----
    'Drying Time',
    'Drying Time Unit',
    'Drying Temperature',
    'Drying Temperature Unit',
]

### Columns by type

UNIT_COLUMNS = [
    "Amount Unit of API",
    "Amount Unit of Coformer",
    "Solvent Amount Unit",
    "Frequency Unit",
    "Mixing Time Unit",
    "Mixing Temperature Unit",
    "Drying Time Unit",
    "Drying Temperature Unit",
]

VARIABLE_COLUMNS = [
    "Type of Synthesis",
    "Solvent",
    "Second Solvent",
    "Third Solvent",
    "Mixing Apparatus",
    "Part of Ratio API",
    "Part of Ratio Coformer",
    "Amount of API",
    "Amount of Coformer",
    "Amount of Solvent",
    "Amount of Second Solvent",
    "Amount of Third Solvent",
    "Mixing Frequency",
    "Mixing Time",
    "Mixing Temperature",
    "Drying Time",
    "Drying Temperature",
]

NUMERIC_COLUMNS = [
    "Part of Ratio API",
    "Part of Ratio Coformer",
    "Amount of API",
    "Amount of Coformer",
    "Amount of Solvent",
    "Amount of Second Solvent",
    "Amount of Third Solvent",
    "Mixing Frequency",
    "Mixing Time",
    "Mixing Temperature",
    "Drying Time",
    "Drying Temperature",
]

CATEGORICAL_COLUMNS = [
    "Type of Synthesis",
    "Solvent",
    "Second Solvent",
    "Third Solvent",
    "Mixing Apparatus",
]

### Columns for experiments

RELEVANT_COLUMNS = [
    "Type of Synthesis",
    "Part of Ratio API",
    "Part of Ratio Coformer",
    "Amount of API",
    "Amount of Coformer",
    "Solvent",
    "Amount of Solvent",
    "Mixing Apparatus",
    "Mixing Frequency",
    "Mixing Time",
    "Mixing Temperature",
]