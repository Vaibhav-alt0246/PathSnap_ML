"""Directed shortest-path utilities for road graphs."""
from __future__ import annotations
import networkx as nx

def shortest_path(graph: nx.MultiDiGraph, source: int, target: int) -> list[int]:
    """Return node IDs for a directed minimum-distance path.

    ``NetworkXNoPath`` is propagated for disconnected or reverse-only paths.
    """
    return nx.shortest_path(graph, source=source, target=target, weight="length_m")

def shortest_path_distance(graph: nx.MultiDiGraph, source: int, target: int) -> float:
    """Return directed shortest-path distance in metres."""
    return float(nx.shortest_path_length(graph, source=source, target=target, weight="length_m"))

def shortest_path_arc_ids(graph: nx.MultiDiGraph, source: int, target: int) -> list[int]:
    """Return original trip-local arc IDs along a shortest node path."""
    nodes = shortest_path(graph, source, target)
    result: list[int] = []
    for start, end in zip(nodes, nodes[1:]):
        options = graph.get_edge_data(start, end)
        if not options:
            raise nx.NetworkXNoPath(f"no edge data for {start}->{end}")
        result.append(int(min(options.values(), key=lambda data: data["length_m"])["arc_id"]))
    return result
