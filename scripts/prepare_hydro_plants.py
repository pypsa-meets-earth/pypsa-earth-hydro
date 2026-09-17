# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Allocates hydro power plants to GloFAS river cells by their upstream area.

Relevant Settings
-----------------

.. code:: yaml

    renewable:
        hydro:
            snapping:
                glofas_uparea:
                merit_root:
                search_radius_km:
                min_upa_km2:
                radius:
                alpha:

Inputs
------

- ``resources/powerplants.csv``: power plant list, see :mod:`build_powerplants`
- ``data/glofas/uparea_glofas_v4_0.nc``: GloFAS v4 upstream area (downloaded if missing)
- ``data/merit_hydro``: MERIT Hydro ``upa`` tiles, extracted or as the original ``upa_*.tar``.
  They require a free registration at https://global-hydrodynamics.github.io/MERIT_Hydro/
  and are therefore not downloaded automatically.

Outputs
-------

- ``resources/hydro_plants.csv``: hydro plants with the allocated GloFAS cell and diagnostics

Description
-----------

GloFAS discharge is only meaningful on the cells of its river network, and the reported
plant coordinate often falls next to the river or on a different branch of it. Each
plant is therefore moved to a nearby GloFAS cell whose upstream area matches the
upstream area expected at the plant:

1. expected upstream area -- read from MERIT Hydro ``upa`` (3 arc-sec, ~90 m) at the
   nearest river cell (``upa >= min_upa_km2``) within ``search_radius_km`` of the plant.
2. allocation -- among the GloFAS cells within ``radius`` cells of the plant, the one
   with the lowest score is chosen::

       S_j = alpha * d_j / d_max + (1 - alpha) * |A_j - A_exp| / A_exp

   with d_j the distance to cell j, d_max the largest distance in the search window,
   A_j the GloFAS upstream area of cell j and A_exp the expected upstream area. The
   relative area mismatch rejects a cell on a much bigger or smaller branch even if it
   is the closest one. Plants without an expected upstream area stay unallocated.
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from _helpers import configure_logging, create_logger, progress_retrieve
from add_electricity import load_powerplants
from pypsa.geo import haversine
from rasterio.windows import Window, from_bounds

logger = create_logger(__name__)

# GloFAS v4 static map, see https://confluence.ecmwf.int/display/CEMS/Auxiliary+Data
GLOFAS_UPAREA_URL = (
    "https://confluence.ecmwf.int/download/attachments/242067380/uparea_glofas_v4_0.nc"
)
MERIT_NODATA = -9999.0  # MERIT `upa` no-data (ocean/undefined)


def load_glofas_uparea(path):
    """
    Open the GloFAS v4 upstream-area map in km2 with ascending x/y coordinates,
    downloading it to `path` if it does not exist.

    Coordinates are rounded to 5 decimals so the cells co-register with the GloFAS
    discharge grid of a cutout.

    Data is licensed under the CEMS-FLOODS licence and should be attributed as
    "Contains modified Copernicus Emergency Management Service information".
    """
    if not os.path.exists(path):
        logger.info(f"Downloading GloFAS uparea map from {GLOFAS_UPAREA_URL} to {path}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        progress_retrieve(GLOFAS_UPAREA_URL, path)

    uparea = xr.open_dataset(path)["uparea"].rename({"longitude": "x", "latitude": "y"})
    uparea = (
        uparea.assign_coords(
            x=np.round(uparea.x.astype(float), 5),
            y=np.round(uparea.y.astype(float), 5),
        )
        .sortby("x")
        .sortby("y")
    )
    return (uparea / 1e6).transpose("y", "x")  # m2 -> km2


def _merit_tile(lat, lon):
    """
    Name of the 5x5 deg MERIT tile containing (lat, lon), e.g. ``n45e010``.
    """
    la = int(np.floor(lat / 5.0) * 5)
    lo = int(np.floor(lon / 5.0) * 5)
    return f"{'s' if la < 0 else 'n'}{abs(la):02d}{'w' if lo < 0 else 'e'}{abs(lo):03d}"


def _merit_package(lat, lon):
    """
    Name of the 30x30 deg tar shipping that tile, e.g. ``upa_n30e000.tar``.
    """
    la = int(np.floor(lat / 30.0) * 30)
    lo = int(np.floor(lon / 30.0) * 30)
    return (
        f"upa_{'s' if la < 0 else 'n'}{abs(la):02d}"
        f"{'w' if lo < 0 else 'e'}{abs(lo):03d}.tar"
    )


def _merit_tiles_for_plants(plants, margin_deg=0.05):
    """
    ``{tile: package}`` of the MERIT tiles covering `plants`, plus a small margin so a
    search window across a tile edge is included.
    """
    tiles = {}
    for lat, lon in zip(plants["lat"].to_numpy(float), plants["lon"].to_numpy(float)):
        for dla in (-margin_deg, margin_deg):
            for dlo in (-margin_deg, margin_deg):
                tiles[_merit_tile(lat + dla, lon + dlo)] = _merit_package(
                    lat + dla, lon + dlo
                )
    return dict(sorted(tiles.items()))


def _merit_tile_path(tile, package, root):
    """
    Locate ``<tile>_upa.tif`` below `root`: extracted, else inside the `package` tar
    (read in place through GDAL's /vsitar). None if unavailable.
    """
    name = f"{tile}_upa.tif"
    hits = sorted(Path(root).rglob(name))
    if hits:
        return str(hits[0])
    for tar in sorted(Path(root).rglob(package)):
        try:
            with tarfile.open(tar) as t:
                member = next((m for m in t.getnames() if m.endswith(name)), None)
        except Exception as e:
            logger.warning(f"Could not read MERIT tar {tar} ({e}).")
            continue
        if member:
            return f"/vsitar/{tar}/{member}"
    return None


def _report_missing_merit(missing, root):
    """
    Tell the user which MERIT packages to download by hand and where to put them.
    """
    logger.error(
        "MERIT Hydro input missing -- tile(s) not found: "
        f"{', '.join(sorted(missing))}\n"
        f"  1. register (free) at https://global-hydrodynamics.github.io/MERIT_Hydro/ "
        "-> password by mail -> open the Dropbox link, folder `v1.0.1/upa`\n"
        f"  2. download: {', '.join(sorted(set(missing.values())))}\n"
        f"  3. put the file(s) in: {root}/\n"
        "     The .tar can stay packed; an extracted <tile>_upa.tif works too.\n"
        "  Plants on a missing tile get no expected upstream area and stay unallocated."
    )


def merit_expected_uparea(plants, merit_root, search_radius_km=1.0, min_upa_km2=10.0):
    """
    Expected upstream area [km2] per plant, read from MERIT Hydro's `upa`.

    Takes the MERIT cells inside a `search_radius_km` box around the plant, keeps those
    on the river network (``upa >= min_upa_km2``) and returns the `upa` of the nearest
    one. The threshold keeps the lookup off the surrounding hillslope cells; windows
    straddling a tile boundary pool the cells of every intersecting tile.

    Returns
    -------
    (pd.Series, pd.Series)
        Expected upstream area [km2] and distance [km] to the MERIT cell it was read
        from; NaN where no river cell was found or the tile is missing.
    """
    area = pd.Series(np.nan, index=plants.index)
    dist = pd.Series(np.nan, index=plants.index)

    # resolve the tiles up front, so missing input is reported in a single message
    needed = _merit_tiles_for_plants(plants)
    paths = {t: _merit_tile_path(t, pkg, merit_root) for t, pkg in needed.items()}
    missing = {t: pkg for t, pkg in needed.items() if paths[t] is None}
    if missing:
        _report_missing_merit(missing, merit_root)
    if not any(paths.values()):
        return area, dist

    srcs = {}
    try:
        for idx, plant in plants.iterrows():
            lon, lat = float(plant["lon"]), float(plant["lat"])
            dlat = search_radius_km / 111.32
            dlon = dlat / max(np.cos(np.radians(lat)), 1e-6)
            bounds = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # w, s, e, n

            a, cx, cy = [], [], []
            for tile in {_merit_tile(y, x) for x in bounds[::2] for y in bounds[1::2]}:
                if paths.get(tile) is None:
                    continue
                if tile not in srcs:
                    srcs[tile] = rasterio.open(paths[tile])
                src = srcs[tile]

                win = from_bounds(*bounds, transform=src.transform)
                r0 = max(int(np.floor(win.row_off)), 0)
                c0 = max(int(np.floor(win.col_off)), 0)
                r1 = min(int(np.ceil(win.row_off + win.height)), src.height)
                c1 = min(int(np.ceil(win.col_off + win.width)), src.width)
                if r1 <= r0 or c1 <= c0:
                    continue
                v = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)

                # MERIT tiles are north-up, so cell centres follow from the transform
                t = src.transform
                gx, gy = np.meshgrid(
                    t.c + (np.arange(c0, c1) + 0.5) * t.a,
                    t.f + (np.arange(r0, r1) + 0.5) * t.e,
                )
                nodata = src.nodata if src.nodata is not None else MERIT_NODATA
                ok = np.isfinite(v) & (v != nodata) & (v >= min_upa_km2)
                if ok.any():
                    a.append(v[ok])
                    cx.append(gx[ok])
                    cy.append(gy[ok])

            if not a:
                continue
            a, cx, cy = map(np.concatenate, (a, cx, cy))
            d = haversine([lon, lat], np.column_stack([cx, cy]))[0]
            j = int(np.argmin(d))
            area.at[idx] = a[j]
            dist.at[idx] = d[j]
    finally:
        for src in srcs.values():
            src.close()

    n = int(area.notna().sum())
    med = np.nanmedian(dist.to_numpy(float)) if n else np.nan
    logger.info(
        f"MERIT uparea: {n}/{len(plants)} plants got an expected upstream area "
        f"(nearest cell with upa >= {min_upa_km2} km2 within {search_radius_km} km; "
        f"median distance to that cell {med:.2f} km)."
    )
    return area, dist


def snap_plants(
    plants,
    glofas_uparea,
    merit_root,
    search_radius_km=1.0,
    min_upa_km2=10.0,
    radius=3,
    alpha=0.5,
):
    """
    Allocate plants to the GloFAS cell that best matches their expected upstream area.

    Parameters
    ----------
    plants : pd.DataFrame
        Plants with `lon`, `lat` columns.
    glofas_uparea : str
        Path to the GloFAS upstream-area map; downloaded if missing.
    merit_root : str
        Directory holding the MERIT `upa` tiles (sub-directories and packed .tar
        files are searched).
    search_radius_km, min_upa_km2 : float
        Lookup of the expected upstream area on the MERIT river network.
    radius : int
        Search radius in GloFAS cells; the window is (2*radius+1)^2 cells.
    alpha : float
        Weight of the distance term; `1 - alpha` weights the area mismatch.

    Returns
    -------
    pd.DataFrame
        Copy of `plants` with `catchment_area` [km2] and `merit_distance_km` from
        MERIT, and `x_snapped`, `y_snapped` (chosen cell centre), `uparea_snapped`
        [km2], `snap_distance` [km] and `snap_score`.
    """
    plants = plants.copy()
    plants["catchment_area"], plants["merit_distance_km"] = merit_expected_uparea(
        plants, merit_root, search_radius_km, min_upa_km2
    )

    area = load_glofas_uparea(glofas_uparea)
    dx = abs(float(area.x[1] - area.x[0]))
    dy = abs(float(area.y[1] - area.y[0]))
    cols = ["x_snapped", "y_snapped", "uparea_snapped", "snap_distance", "snap_score"]
    for col in cols:
        plants[col] = np.nan

    for idx, plant in plants[["lon", "lat", "catchment_area"]].iterrows():
        lon, lat, a_exp = plant.astype(float)
        if not np.isfinite(a_exp) or a_exp <= 0:
            continue

        win = area.sel(
            x=slice(lon - (radius + 0.5) * dx, lon + (radius + 0.5) * dx),
            y=slice(lat - (radius + 0.5) * dy, lat + (radius + 0.5) * dy),
        )
        cx, cy = np.meshgrid(win.x.values, win.y.values)
        a, cx, cy = win.values.ravel(), cx.ravel(), cy.ravel()
        ok = np.isfinite(a)
        if not ok.any():
            continue
        a, cx, cy = a[ok], cx[ok], cy[ok]

        d = haversine([lon, lat], np.column_stack([cx, cy]))[0]
        score = alpha * d / max(d.max(), 1e-9) + (1 - alpha) * np.abs(a - a_exp) / a_exp
        j = int(np.argmin(score))
        plants.loc[idx, cols] = (cx[j], cy[j], a[j], d[j], score[j])

    n = int(plants["snap_score"].notna().sum())
    logger.info(f"Upstream-area snapping: {n}/{len(plants)} plants allocated.")
    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("prepare_hydro_plants")

    configure_logging(snakemake)

    ppls = load_powerplants(snakemake.input.powerplants)
    hydro_ppls = ppls[ppls.carrier == "hydro"]

    snapping = {k: v for k, v in snakemake.params.snapping.items() if k != "enable"}
    snap_plants(hydro_ppls, **snapping).to_csv(snakemake.output.hydro_plants)
