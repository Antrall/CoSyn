from __future__ import annotations
from typing import Callable, FrozenSet, Iterable, Set
from tqdm.auto import tqdm

import pandas as pd

from chow_liu import ChowLiuTree, MutualInfo
from bayes_network import BayesNetwork

Edge = FrozenSet[str]
EdgeFitFn = Callable[[pd.DataFrame], Set[Edge]]


def chow_liu_edge_fn(columns: Iterable[str], min_pairs: int = 5) -> EdgeFitFn:
    """
    fit_fn: trains Chow-Liu tree on columns and returns its edges as frozensets
    """
    columns = list(columns)
    mi_estimator = MutualInfo(min_pairs=min_pairs)

    def fit_fn(df: pd.DataFrame) -> Set[Edge]:
        mi_result = mi_estimator.compute(df, columns=columns)
        tree = ChowLiuTree().fit(mi_result.info_matrix)
        return {frozenset(edge) for edge in tree.graph.edges()}

    return fit_fn


def bayes_network_edge_fn(columns: Iterable[str], criterion: str, max_parents: int = 1, max_iterations: int = 100) -> EdgeFitFn:
    """
    fit_fn: trains Bayes Network on columns and returns its edges as frozensets
    """
    columns = list(columns)

    def fit_fn(df: pd.DataFrame) -> Set[Edge]:
        network = BayesNetwork(max_parents=max_parents, max_iterations=max_iterations, criterion=criterion)
        network.fit(df, columns)
        return {frozenset(edge) for edge in network.graph.edges()}

    return fit_fn


def hillclimb_network_edge_fn(columns: Iterable[str], criterion: str, max_parents: int = 1) -> EdgeFitFn:
    """
    fit_fn: trains HillClimb Search on columns and returns its edges as frozensets
    """
    columns = list(columns)

    def fit_fn(df: pd.DataFrame) -> Set[Edge]:
        from pgmpy.causal_discovery import HillClimbSearch
        network = HillClimbSearch(scoring_method=criterion, max_indegree=max_parents)
        network.fit(df[columns])
        return {frozenset(edge) for edge in network.causal_graph_.to_undirected().edges()}

    return fit_fn


class NetworkBootstrap:
    """
    Bootstrap resampling and structure retraining to estimate stability of edges.
    """
    def __init__(self, fit_fn: EdgeFitFn, n_iterations: int = 50, random_state: int = None):
        self.fit_fn = fit_fn
        self.n_iterations = n_iterations
        self.random_state = random_state

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        counts: dict[Edge, int] = {}

        for i in tqdm(range(self.n_iterations)):
            sample = df.sample(n=len(df), replace=True, random_state=self._seed(i))
            edges = self.fit_fn(sample)
            for edge in edges:
                counts[edge] = counts.get(edge, 0) + 1

        rows = [
            {"V1": sorted(edge)[0], "V2": sorted(edge)[1], "frequency": count / self.n_iterations}
            for edge, count in counts.items()
        ]
        return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)

    def _seed(self, i: int):
        if self.random_state is None:
            return None
        return self.random_state + i