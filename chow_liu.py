from __future__ import annotations
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import networkx as nx

from resources.schemas.config import *
from sklearn.metrics import mutual_info_score


class MutualInfo:
    """
    Computes:
    - info_matrix: symmetric matrix of pairwise mutual information scores
    - count_matrix: number of jointly-observed rows for each pair
    """
    def __init__(self, min_pairs: int = 5):
        self.min_pairs = min_pairs
        self.info_matrix = None
        self.count_matrix = None

    def compute(self, df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> "MutualInfo":
        columns = list(columns) if columns is not None else list(df.columns)
        n = len(columns)

        info_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
        count_matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=int)

        for i in range(n):
            col_i = columns[i]
            info_matrix.loc[col_i, col_i] = np.nan
            count_matrix.loc[col_i, col_i] = int(df[col_i].notna().sum())

            for j in range(i + 1, n):
                col_j = columns[j]
                pair = df[[col_i, col_j]].dropna()
                n_pairs = len(pair.copy().replace(MISSING_MARKER, pd.NA).dropna())
                count_matrix.loc[col_i, col_j] = count_matrix.loc[col_j, col_i] = n_pairs

                if n_pairs < self.min_pairs:
                    continue

                score = mutual_info_score(pair[col_i], pair[col_j])
                info_matrix.loc[col_i, col_j] = info_matrix.loc[col_j, col_i] = score

        self.info_matrix = info_matrix
        self.count_matrix = count_matrix
        return self

    def top_pairs(self, k: int = 10) -> pd.DataFrame:
        pairs = []
        cols = list(self.info_matrix.columns)
        for i, col_i in enumerate(cols):
            for col_j in cols[i + 1 :]:
                score = self.info_matrix.loc[col_i, col_j]

                if pd.isna(score):
                    continue

                pairs.append({
                    "V1": col_i,
                    "V2": col_j,
                    "mutual_info": score,
                    "n_pairs": int(self.count_matrix.loc[col_i, col_j]),
                })
        
        result = pd.DataFrame(pairs).sort_values("mutual_info", ascending=False)
        return result.head(k).reset_index(drop=True)


class ChowLiuTree:
    """
    Probabilistic graphical model Chow Liu tree:
    - Training: on the information matrix between all variables
    - Structure: MST from full graph, weights based from MI matrix
    - Orientation: from any vertex designated as the root
    """
    def __init__(self):
        self.graph: Optional[nx.Graph] = None

    def fit(self, mi_matrix: pd.DataFrame) -> "ChowLiuTree":
        full_graph = nx.Graph()
        full_graph.add_nodes_from(mi_matrix.columns)

        cols = list(mi_matrix.columns)
        for i, col_i in enumerate(cols):
            for col_j in cols[i + 1 :]:
                weight = mi_matrix.loc[col_i, col_j]
                if pd.isna(weight):
                    continue
                full_graph.add_edge(col_i, col_j, weight=weight)

        self.graph = nx.maximum_spanning_tree(full_graph, weight="weight")
        return self

    def edge_table(self) -> pd.DataFrame:
        rows = [{"V1": u, "V2": v, "mutual_info": data["weight"]} for u, v, data in self.graph.edges(data=True)]
        return pd.DataFrame(rows).sort_values("mutual_info", ascending=False).reset_index(drop=True)

    def to_directed(self, root: str) -> nx.DiGraph:
        return nx.bfs_tree(self.graph, source=root)