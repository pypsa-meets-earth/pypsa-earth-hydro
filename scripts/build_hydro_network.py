# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Builds a water network of the rivers in the modelled region from GloFAS river
discharge; all flows are in m3/s.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    renewable:
        hydro:
            cutout:

    hydro_network:
        ldd:
        uparea:
        min_upstream_area_km2:
        bus_spacing_km:
        clip_negative_inflow:

Inputs
------

- ``resources/shapes/country_shapes.geojson``: the network covers their bounding box, see :mod:`build_shapes`
- ``"cutouts/" + config["renewable"]["hydro"]["cutout"]``: cutout with GloFAS ``discharge``, see :mod:`build_cutout`
- ``data/glofas/ldd_glofas_v4_0.nc``, ``data/glofas/uparea_glofas_v4_0.nc``: GloFAS v4 local drainage direction and upstream area (downloaded if missing)

Outputs
-------

- ``networks/hydro_network.nc``: water network of buses, links and generators carrying m3/s

Description
-----------

GloFAS discharge carries no flow direction, so the rivers and their topology are
rebuilt from the GloFAS static maps within the bounding box of the countries, clipped
to the cutout:

1. A GloFAS cell is a river cell if it has a drainage direction and an upstream area
   above ``min_upstream_area_km2``. The local drainage direction links each river cell
   to its downstream river cell, which turns the rivers into a forest of trees.
2. Buses are placed at headwaters, confluences and outlets, and along each reach
   wherever the river length since the previous bus reaches ``bus_spacing_km``. A link
   connects each bus to the next bus downstream and carries water downstream only.
3. A generator at each bus injects the fixed local inflow: the discharge at the bus
   minus the discharge at the buses directly upstream, with negative values set to
   zero if ``clip_negative_inflow``.
4. A sink generator at each outlet takes the water leaving the region.

Link and sink capacities are sized from the flow accumulated downstream, so the flows
follow directly from the injections. Clipping negative inflow adds water; the
resulting deviation of the accumulated flow from the discharge is logged.
"""

import atlite
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, create_logger
from prepare_hydro_plants import load_glofas_map, load_glofas_uparea
from pypsa.geo import haversine_pts

logger = create_logger(__name__)

# PCRaster LDD code -> (row, col) step to the downstream cell on a grid with ascending
# y, i.e. rows run northward: 1 = SW, 2 = S, ..., 9 = NE; 5 is a pit (outlet)
LDD_DELTA = {
    1: (-1, -1),
    2: (-1, 0),
    3: (-1, 1),
    4: (0, -1),
    6: (0, 1),
    7: (1, -1),
    8: (1, 0),
    9: (1, 1),
}


def river_cells(ldd, uparea, min_upstream_area_km2):
    """
    Find the river cells and the downstream river cell of each.

    Cells are flat indices into the `ldd` grid (``row * nx + col``).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Flat indices of the river cells, and the downstream river cell per grid
        cell; -1 for outlets (pit, or draining out of the grid or into a non-river
        cell) and non-river cells.
    """
    ny, nx = ldd.shape
    code = np.nan_to_num(ldd.values).astype(int).ravel()
    is_river = (
        (code >= 1) & (code <= 9) & (uparea.values.ravel() > min_upstream_area_km2)
    )

    row, col = np.divmod(np.arange(ny * nx), nx)
    for k, (dr, dc) in LDD_DELTA.items():
        row[code == k] += dr
        col[code == k] += dc
    inside = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    target = np.where(inside, row * nx + col, 0)
    down = np.where(is_river & inside & (code != 5) & is_river[target], target, -1)
    return np.flatnonzero(is_river), down


def place_buses(river, down, x, y, spacing_km):
    """
    Place buses on the river cells and link each to the next bus downstream.

    Buses sit at headwaters, confluences and outlets, which preserves the topology,
    and along each reach wherever the river length since the previous bus reaches
    `spacing_km`.

    Returns
    -------
    (np.ndarray, pd.DataFrame)
        Bus cells, and links with the columns `bus0` (upstream cell), `bus1`
        (downstream cell) and `length` [km].
    """
    has_down = down >= 0
    step = np.zeros(len(down))
    step[has_down] = haversine_pts(
        np.c_[x[has_down], y[has_down]],
        np.c_[x[down[has_down]], y[down[has_down]]],
    )
    indeg = np.bincount(down[has_down], minlength=len(down))

    # headwaters (indeg 0), confluences (indeg >= 2) and outlets
    fixed = np.zeros(len(down), dtype=bool)
    fixed[river[(indeg[river] != 1) | ~has_down[river]]] = True

    # walk each reach from its headwater or confluence down to the next fixed bus
    links = []
    for start in river[(indeg[river] != 1) & has_down[river]]:
        bus, cell, length = start, down[start], step[start]
        while not fixed[cell]:
            if length >= spacing_km:
                links.append((bus, cell, length))
                bus, length = cell, 0.0
            length += step[cell]
            cell = down[cell]
        links.append((bus, cell, length))

    links = pd.DataFrame(links, columns=["bus0", "bus1", "length"])
    return np.union1d(np.flatnonzero(fixed), links.bus1), links


def build_hydro_network(
    ldd,
    uparea,
    cutout,
    snapshots,
    min_upstream_area_km2=500.0,
    bus_spacing_km=50.0,
    clip_negative_inflow=True,
):
    """
    Build the water network of the rivers on the `ldd` grid.

    Parameters
    ----------
    ldd, uparea : xr.DataArray
        GloFAS local drainage direction and upstream area [km2] on the same grid
        with ascending x/y.
    cutout : atlite.Cutout
        Cutout holding the GloFAS ``discharge``.
    snapshots : pd.DatetimeIndex
        Snapshots of the network; the discharge is interpolated onto them.
    min_upstream_area_km2 : float
        Upstream area above which a cell belongs to the river network.
    bus_spacing_km : float
        River length after which a bus is placed along a reach.
    clip_negative_inflow : bool
        Set negative local inflow to zero.

    Returns
    -------
    pypsa.Network
    """
    river, down = river_cells(ldd, uparea, min_upstream_area_km2)
    x, y = (a.ravel() for a in np.meshgrid(ldd.x.values, ldd.y.values))
    cells, links = place_buses(river, down, x, y, bus_spacing_km)

    buses = pd.DataFrame(
        {"x": x[cells], "y": y[cells], "uparea_km2": uparea.values.ravel()[cells]},
        index="hbus_" + pd.Index(cells).astype(str),
    )
    links.index = "hlink_" + links.bus0.astype(str) + "_" + links.bus1.astype(str)
    links[["bus0", "bus1"]] = "hbus_" + links[["bus0", "bus1"]].astype(str)

    # each river is a tree, identified by its outlet bus
    graph = nx.DiGraph()
    graph.add_nodes_from(buses.index)
    graph.add_edges_from(zip(links.bus0, links.bus1))
    outlets = buses.index.difference(links.bus0)
    river_id = {b: o for o in outlets for b in nx.ancestors(graph, o) | {o}}
    buses["river_id"] = buses.index.map(river_id)
    logger.info(
        f"Placed {len(buses)} buses and {len(links)} links on {len(outlets)} rivers."
    )

    plants = buses[["x", "y"]].rename(columns={"x": "lon", "y": "lat"})
    discharge = (
        cutout.hydro(plants.rename_axis("plant"), module="glofas", time=snapshots)
        .to_pandas()
        .T.astype(float)
    )

    # local inflow = discharge minus the discharge flowing in from the upstream buses
    upstream = discharge[links.bus0].T.groupby(links.bus1.values).sum().T
    local = discharge - upstream.reindex(columns=discharge.columns, fill_value=0.0)
    if clip_negative_inflow:
        local = local.clip(lower=0.0)

    # flow leaving each bus = local inflow accumulated downstream
    flow = local.copy()
    for bus in nx.topological_sort(graph):
        for bus1 in graph.successors(bus):
            flow[bus1] += flow[bus]
    residual = (flow - discharge).abs().mean().sum() / discharge.mean().sum()
    logger.info(
        f"The accumulated local inflow deviates by {residual:.2%} from the discharge "
        "(mean absolute deviation; clipping negative inflow adds water)."
    )
    # capacities: peak flow plus headroom, so the fixed flows always fit
    peak = flow.max() * 1.2 + 1.0

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Carrier", "water")
    n.madd(
        "Bus",
        buses.index,
        x=buses.x,
        y=buses.y,
        carrier="water",
        unit="m3/s",
        river_id=buses.river_id,
        uparea_km2=buses.uparea_km2,
    )
    # links carry water downstream only (default p_min_pu = 0); the small cost only
    # gives the optimisation an objective, the flows are fixed by the inflow
    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        carrier="water",
        length=links.length,
        p_nom=peak[links.bus0].values,
        marginal_cost=1e-3,
    )
    p_nom = local.max().clip(lower=1e-6)
    p_pu = (local / p_nom).clip(upper=1.0)
    n.madd(
        "Generator",
        buses.index,
        suffix=" inflow",
        bus=buses.index,
        carrier="water",
        p_nom=p_nom,
        p_min_pu=p_pu,
        p_max_pu=p_pu,
    )
    n.madd(
        "Generator",
        outlets,
        suffix=" spill",
        bus=outlets,
        carrier="water",
        p_nom=peak[outlets],
        p_min_pu=-1.0,
        p_max_pu=0.0,
    )
    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_hydro_network")

    configure_logging(snakemake)

    params = snakemake.params.hydro_network
    cutout = atlite.Cutout(snakemake.input.cutout)

    # atlite reads the discharge at the nearest cutout cell without a distance limit,
    # so the region is clipped to the cutout
    x0, y0, x1, y1 = gpd.read_file(snakemake.input.country_shapes).total_bounds
    cx0, cy0, cx1, cy1 = cutout.bounds
    x0, y0, x1, y1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)

    ldd = load_glofas_map(params["ldd"], "ldd").sel(x=slice(x0, x1), y=slice(y0, y1))
    uparea = load_glofas_uparea(params["uparea"]).sel(x=ldd.x, y=ldd.y)

    n = build_hydro_network(
        ldd,
        uparea,
        cutout,
        pd.date_range(freq="h", **snakemake.params.snapshots),
        min_upstream_area_km2=params["min_upstream_area_km2"],
        bus_spacing_km=params["bus_spacing_km"],
        clip_negative_inflow=params["clip_negative_inflow"],
    )
    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output.hydro_network)
