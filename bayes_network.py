from __future__ import annotations
from typing import Iterable, Tuple

import itertools
import networkx as nx
import numpy as np
import pandas as pd

CRITERION_PENALTY = {
    "bic": lambda n: 0.5 * np.log(n),
    "aic": lambda n: 1.0,
}


def criterion(df: pd.DataFrame, node: str, parents: Tuple[str, ...], criterion: str = "bic") -> float:
    """
    Score for one node with a given set of parents. Supports different penalties and criterions
    """
    cols = [node] + list(parents)
    complete = df[cols].dropna()
    n = len(complete)

    if n == 0:
        return -np.inf

    if parents:
        log_lik = 0.0
        for _, group in complete.groupby(list(parents), observed=True)[node]:
            counts = group.value_counts()
            probs = counts / counts.sum()
            log_lik += (counts * np.log(probs)).sum()
        n_parent_configs = complete[list(parents)].drop_duplicates().shape[0]
    else:
        counts = complete[node].value_counts()
        probs = counts / counts.sum()
        log_lik = (counts * np.log(probs)).sum()
        n_parent_configs = 1

    n_node_values = complete[node].nunique()
    n_free_params = n_parent_configs * max(n_node_values - 1, 0)
    penalty = n_free_params * CRITERION_PENALTY[criterion](n)
    return float(log_lik - penalty)


class BayesNetwork:
    """
    Greedy search for grapth structure:
    - Maximizes the information criterion, counts it for a node and its parents
    - According to criterion diff, adds or removes edges
    """
    def __init__(self, max_parents: int = 3, max_iterations: int = 200, criterion: str = "bic"):
        self.max_parents = max_parents
        self.max_iterations = max_iterations
        self.criterion = criterion
        self.graph = None

    def fit(self, df: pd.DataFrame, columns: Iterable[str]) -> "BayesNetwork":
        columns = list(columns)
        graph = nx.DiGraph()
        graph.add_nodes_from(columns)

        node_scores = {node: criterion(df, node, (), self.criterion) for node in columns}

        for _ in range(self.max_iterations):
            best_delta = 0.0
            best_optim = None

            for u, v in itertools.permutations(columns, 2):
                parents_v = tuple(graph.predecessors(v))

                if not graph.has_edge(u, v) and len(parents_v) < self.max_parents:
                    if not self._is_cycle(graph, u, v):
                        candidate_parents = tuple(sorted(parents_v + (u,)))
                        new_score = criterion(df, v, candidate_parents, self.criterion)
                        delta = new_score - node_scores[v]
                        if delta > best_delta:
                            best_delta = delta
                            best_optim = ("add", u, v, candidate_parents, new_score)

                if graph.has_edge(u, v):
                    candidate_parents = tuple(sorted(p for p in parents_v if p != u))
                    new_score = criterion(df, v, candidate_parents, self.criterion)
                    delta = new_score - node_scores[v]
                    if delta > best_delta:
                        best_delta = delta
                        best_optim = ("remove", u, v, candidate_parents, new_score)

            if best_optim is None:
                break

            move, u, v, new_parents, new_score = best_optim
            if move == "add":
                graph.add_edge(u, v)
            else:
                graph.remove_edge(u, v)
            node_scores[v] = new_score

        self.graph = graph
        return self

    @staticmethod
    def _is_cycle(graph: nx.DiGraph, u: str, v: str) -> bool:
        graph.add_edge(u, v)
        has_cycle = not nx.is_directed_acyclic_graph(graph)
        graph.remove_edge(u, v)
        return has_cycle

    def edge_table(self) -> pd.DataFrame:
        rows = [{"parent": u, "child": v} for u, v in self.graph.edges()]
        return pd.DataFrame(rows)