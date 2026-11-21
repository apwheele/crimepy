"""Bundled datasets the library loads at runtime."""

import crimepy.cdcplot as cdcplot
from crimepy.data import (
    load_carrollton_data,
    load_dallas_border,
    load_dallas_data,
    load_network_data,
)


def test_logo_image_loads():
    assert cdcplot.im.ndim >= 2
    assert cdcplot.im.size > 0


def test_dallas_incidents_load():
    df = load_dallas_data()
    assert len(df) > 0
    assert "geometry" in df.columns
    assert df.crs is not None
    assert {"begin", "end", "NIBR_DESC", "LOC_DESC"} <= set(df.columns)


def test_dallas_border_loads():
    border = load_dallas_border()
    assert len(border) > 0
    assert "geometry" in border.columns
    assert border.geometry.notna().all()


def test_gang_network_loads():
    graph, nodes, edges = load_network_data()
    assert graph.number_of_nodes() == len(nodes)
    assert graph.number_of_edges() > 0
    assert len(edges) > 0
    assert "Id" in nodes.columns


def test_carrollton_grid_loads():
    grid = load_carrollton_data()
    assert len(grid) > 0
    assert "geometry" in grid.columns
    assert grid.crs is not None
