import networkx as nx
import pandas as pd
import pytest

from src.data.loader import TripData
from src.graph.road_graph import build_road_graph
from src.graph.shortest_path import shortest_path, shortest_path_arc_ids, shortest_path_distance

def trip(nodes, arcs):
    return TripData("synthetic", pd.DataFrame(columns=["longitude", "latitude", "timestamp"]), pd.DataFrame(nodes, columns=["longitude", "latitude"]), pd.DataFrame(arcs, columns=["from_node", "to_node"]), route=pd.Series([999]))

def test_directed_graph_and_metric_length():
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0)], [(0, 1)]))
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.nodes[0]["longitude"] == 0.0
    assert graph[0][1][0]["arc_id"] == 0
    assert 100 < graph[0][1][0]["length_m"] < 120

def test_reverse_direction_is_unreachable():
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0)], [(0, 1)]))
    with pytest.raises(nx.NetworkXNoPath):
        shortest_path(graph, 1, 0)

def test_parallel_arcs_preserve_original_ids():
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0)], [(0, 1), (0, 1)]))
    assert set(graph[0][1]) == {0, 1}
    assert {data["arc_id"] for data in graph[0][1].values()} == {0, 1}

def test_disconnected_graph_has_no_path_and_distance():
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)], [(0, 1)]))
    with pytest.raises(nx.NetworkXNoPath):
        shortest_path_distance(graph, 0, 2)

def test_shortest_path_maps_back_to_arc_ids():
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)], [(0, 1), (1, 2)]))
    assert shortest_path(graph, 0, 2) == [0, 1, 2]
    assert shortest_path_arc_ids(graph, 0, 2) == [0, 1]

def test_invalid_node_reference_is_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        build_road_graph(trip([(0.0, 0.0)], [(0, 3)]))

@pytest.mark.parametrize("endpoint", [1.5, float("nan"), float("inf"), "not-a-node"])
def test_malformed_or_fractional_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError, match="malformed"):
        build_road_graph(trip([(0.0, 0.0), (0.001, 0.0)], [(0, endpoint)]))

@pytest.mark.parametrize("endpoint", [1.0, "1", " 1 "])
def test_integer_valued_endpoint_is_accepted(endpoint):
    graph = build_road_graph(trip([(0.0, 0.0), (0.001, 0.0)], [(0, endpoint)]))
    assert graph[0][1][0]["to_node"] == 1
