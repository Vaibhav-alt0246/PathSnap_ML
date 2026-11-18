"""Log-space HMM/Viterbi decoding over candidate road arcs."""
from __future__ import annotations

import math
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.graph.road_graph import build_road_graph
from src.graph.shortest_path import shortest_path_distance


def _sparse_transition_graph(graph: nx.MultiDiGraph):
    """Build an exact directed CSR graph, keeping the shortest parallel arc."""
    n = graph.number_of_nodes()
    weights = {}
    for source, target, data in graph.edges(data=True):
        key = (int(source), int(target))
        value = float(data["length_m"])
        weights[key] = min(value, weights.get(key, math.inf))
    if not weights:
        return csr_matrix((n, n), dtype=float)
    rows, cols, vals = zip(*[(u, v, w) for (u, v), w in weights.items()])
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)


def decode(trip: object, candidates: pd.DataFrame, probabilities: Sequence[float], *, use_sparse: bool = True) -> list[int]:
    """Return the most likely ordered trip-local arc IDs.

    Candidate rows and probabilities must have identical order. Emissions are
    supplied classifier probabilities in log space. Transitions are feasible
    only when directed graph connectivity exists between consecutive arcs.
    """
    required = {"point_idx", "arc_id", "projection_fraction"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    probs = np.asarray(probabilities, dtype=float)
    if len(probs) != len(candidates) or not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite and match candidate row order")
    graph = build_road_graph(trip)
    endpoints = {int(data["arc_id"]): (int(source), int(target))
                 for source, target, _, data in graph.edges(keys=True, data=True)}
    grouped = [(int(point), frame) for point, frame in candidates.groupby("point_idx", sort=True)]
    if not grouped:
        return []
    log_emissions = np.log(np.clip(probs, 1e-300, 1.0))
    row_positions = {index: position for position, index in enumerate(candidates.index)}
    states = [list(frame.index) for _, frame in grouped]
    scores = {index: float(log_emissions[row_positions[index]]) for index in states[0]}
    backpointers: list[dict[int, int | None]] = [{index: None for index in states[0]}]
    distance_cache = {}
    sparse_graph = _sparse_transition_graph(graph) if use_sparse else None
    for current_states, previous_states in zip(states[1:], states[:-1]):
        next_scores: dict[int, float] = {}
        pointers: dict[int, int | None] = {}
        for current in current_states:
            curr = candidates.loc[current]
            best_score, best_previous = -math.inf, None
            curr_start = endpoints[int(curr["arc_id"])][0]
            for previous in previous_states:
                if not math.isfinite(scores.get(previous, -math.inf)):
                    continue
                prev = candidates.loc[previous]
                prev_end = endpoints[int(prev["arc_id"])][1]
                if prev_end not in distance_cache:
                    if sparse_graph is not None:
                        distance_cache[prev_end] = dijkstra(sparse_graph, directed=True, indices=prev_end)
                    else:
                        distance_cache[prev_end] = nx.single_source_dijkstra_path_length(graph, prev_end, weight="length_m")
                distances = distance_cache[prev_end]
                distance = float(distances[curr_start])
                if not math.isfinite(distance):
                    continue
                candidate_score = scores[previous] - distance + float(log_emissions[row_positions[current]])
                if candidate_score > best_score:
                    best_score, best_previous = candidate_score, previous
            next_scores[current] = best_score
            pointers[current] = best_previous
        if not any(math.isfinite(value) for value in next_scores.values()):
            # No directed continuation exists; use the best emission per
            # observation rather than inventing a graph transition.
            best = max(current_states, key=lambda index: log_emissions[row_positions[index]])
            next_scores[best] = float(log_emissions[row_positions[best]])
            pointers[best] = None
        scores, backpointers = next_scores, backpointers + [pointers]
    final = max(scores, key=lambda index: (scores[index], -int(candidates.loc[index, "arc_id"])))
    selected = [final]
    for pointers in reversed(backpointers[1:]):
        previous = pointers.get(selected[-1])
        if previous is None:
            return [int(candidates.loc[index, "arc_id"]) for index in [
                max(frame.index, key=lambda item: log_emissions[row_positions[item]])
                for _, frame in grouped
            ]]
        selected.append(previous)
    return [int(candidates.loc[index, "arc_id"]) for index in reversed(selected)]


def _arc_endpoints(graph: nx.MultiDiGraph, arc_id: int) -> tuple[int, int]:
    """Find endpoints for a trip-local arc ID."""
    for source, target, _, data in graph.edges(keys=True, data=True):
        if int(data["arc_id"]) == arc_id:
            return int(source), int(target)
    raise KeyError(f"arc_id {arc_id} is not present in graph")
