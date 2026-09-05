"""Manuscript Figure 4: structural exposure of European rating schemes.

The map combines the 2015-2023 decline in electricity carbon intensity with the
country typology written by :mod:`epc_artefact.typology`. It deliberately contains
one panel: the manuscript's map. England and Wales, rather than the whole United
Kingdom, receive the empirical-case hatch when Natural Earth subunits are available.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import (
    NE_COUNTRIES_GEOJSON,
    NE_COUNTRIES_GEOJSON_URL,
    NE_SUBUNITS_GEOJSON,
    NE_SUBUNITS_GEOJSON_URL,
    OUT_FIGURES,
    TYPOLOGY_END_YEAR,
    TYPOLOGY_START_YEAR,
    ensure_dirs,
)
from .data import download_if_missing
from .typology import EUROPE_SCOPE, build_typology, load_owid

Y0, Y1 = TYPOLOGY_START_YEAR, TYPOLOGY_END_YEAR
MAP_VMAX = 70
METRIC_COLOURS = {
    "Carbon-weighted": "#D323C8",
    "Dual / CO2 reported": "#4AA8A3",
    "Primary energy": "#17395E",
    "Delivered / final energy": "#6AA84F",
    "Not classified": "#BDBDBD",
}
METRIC_LABELS = {
    "Carbon-weighted": "Carbon-Weighted",
    "Dual / CO2 reported": "Dual/CO$_2$ shown",
    "Primary energy": "Primary Energy",
    "Delivered / final energy": "Delivered Energy",
    "Not classified": "Not Classified",
}
THRESH_MARKERS = {"current": "s", "planned": "^", "none": "o", "unknown": "x"}
THRESH_LABELS = {
    "current": "Current Threshold",
    "planned": "Planned Threshold",
    "none": "Non-Audited",
    "unknown": "Unknown",
}
NAME_FIX = {
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "Czech Rep.": "Czechia",
    "Macedonia": "North Macedonia",
}


def _read_geodata(gpd, local_path, fallback):
    """Read cached boundaries, downloading atomically when the cache is absent."""
    if not local_path.exists():
        download_if_missing(fallback, local_path)
    try:
        return gpd.read_file(local_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read cached boundary data at {local_path}") from error


def _country_name_column(frame) -> str:
    for candidate in ("NAME", "ADMIN", "name", "admin"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Natural Earth country boundaries have no recognized name column")


def _mainland_point(geometry):
    """Place a marker on the largest polygon, avoiding remote-island centroids."""
    parts = list(getattr(geometry, "geoms", ()))
    if parts:
        geometry = max(parts, key=lambda part: part.area)
    return geometry.representative_point()


def make_map() -> None:
    """Write the single-panel, manuscript-style Figure 4 as PDF and PNG."""
    import geopandas as gpd

    ensure_dirs()
    owid = load_owid()
    typology = build_typology()
    scope = [country for country in EUROPE_SCOPE if country != "European Union (27)"]
    value_column = next(column for column in owid.columns if "intensity" in column.lower())
    grid = (
        owid[owid.Entity.isin(scope) & owid.Year.isin([Y0, Y1])]
        .pivot_table(index="Entity", columns="Year", values=value_column)
        .reset_index()
        .rename(columns={"Entity": "country", Y0: "grid_start", Y1: "grid_end"})
    )
    grid["decline_pct"] = 100 * (grid.grid_start - grid.grid_end) / grid.grid_start

    world = _read_geodata(gpd, NE_COUNTRIES_GEOJSON, NE_COUNTRIES_GEOJSON_URL)
    name_column = _country_name_column(world)
    world["plot_country"] = world[name_column].replace(NAME_FIX)
    europe = world[world.plot_country.isin(scope)].copy()
    europe = europe.merge(grid[["country", "decline_pct"]], how="left",
                          left_on="plot_country", right_on="country")
    europe = europe.merge(
        typology[["country", "metric_class", "threshold_status"]],
        how="left", left_on="plot_country", right_on="country", suffixes=("", "_typology"),
    )
    if europe.empty:
        raise ValueError("No European countries matched the Natural Earth boundaries")

    points = europe.copy()
    points["geometry"] = points.geometry.map(_mainland_point)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(9.0, 8.6))
    fig.subplots_adjust(left=0.015, right=0.76, top=0.985, bottom=0.02)

    europe.plot(
        ax=ax,
        column="decline_pct",
        cmap="YlOrBr",
        vmin=0,
        vmax=MAP_VMAX,
        linewidth=0.45,
        edgecolor="#B9B9B9",
        missing_kwds={"color": "#F1F1F1", "edgecolor": "#C8C8C8", "hatch": "///"},
        zorder=1,
    )
    europe.boundary.plot(ax=ax, linewidth=0.40, color="#B0B0B0", zorder=2)

    # Restrict the empirical-case hatch to the two register jurisdictions.
    try:
        subunits = _read_geodata(gpd, NE_SUBUNITS_GEOJSON, NE_SUBUNITS_GEOJSON_URL)
        subunit_column = next(
            column for column in ("SUBUNIT", "GEOUNIT", "NAME", "name")
            if column in subunits.columns
        )
        empirical = subunits[subunits[subunit_column].isin(["England", "Wales"])]
        if empirical.empty:
            raise ValueError("England and Wales were not found in Natural Earth subunits")
        empirical.plot(ax=ax, facecolor="none", edgecolor="#D5670B", linewidth=1.0,
                       hatch="////", zorder=4)
    except (OSError, ValueError, StopIteration):
        # Boundaries may be unavailable offline. A UK outline is an explicit visual
        # fallback, not a change to the country-level metric or plotted data.
        empirical = europe[europe.plot_country == "United Kingdom"]
        empirical.plot(ax=ax, facecolor="none", edgecolor="#D5670B", linewidth=1.0,
                       hatch="////", zorder=4)

    for _, row in points.dropna(subset=["metric_class"]).iterrows():
        marker = THRESH_MARKERS.get(row.threshold_status, "x")
        colour = METRIC_COLOURS.get(row.metric_class, METRIC_COLOURS["Not classified"])
        kwargs = {"marker": marker, "s": 88, "color": colour,
                  "linewidth": 0.9, "zorder": 5}
        if marker == "x":
            kwargs["color"] = "#222222"
        else:
            kwargs["edgecolor"] = "white"
        ax.scatter(row.geometry.x, row.geometry.y, **kwargs)

    ax.set_xlim(-12, 35)
    ax.set_ylim(34, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    metric_order = ["Carbon-weighted", "Dual / CO2 reported", "Primary energy"]
    # Retain a separate delivered-energy key if the current typology contains it;
    # otherwise follow the four-key manuscript legend exactly.
    if (points.metric_class == "Delivered / final energy").any():
        metric_order.append("Delivered / final energy")
    metric_order.append("Not classified")
    metric_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=9,
               markerfacecolor=METRIC_COLOURS[key], markeredgecolor="none",
               label=METRIC_LABELS[key])
        for key in metric_order
    ]
    metric_legend = ax.legend(
        handles=metric_handles, title="Metric Class", loc="upper left",
        bbox_to_anchor=(1.01, 0.99), frameon=False, borderaxespad=0,
        fontsize=9.3, title_fontsize=10.2, handlelength=1.0, handletextpad=0.5,
    )
    ax.add_artist(metric_legend)

    threshold_handles = []
    for key in ("current", "planned", "none", "unknown"):
        marker = THRESH_MARKERS[key]
        threshold_handles.append(Line2D(
            [0], [0], marker=marker, linestyle="none", markersize=9,
            markerfacecolor="white" if marker != "x" else "none",
            markeredgecolor="#222222", color="#222222", label=THRESH_LABELS[key],
        ))
    threshold_handles.append(Patch(facecolor="white", edgecolor="#D5670B", hatch="////",
                                   label="Empirical case"))
    ax.legend(
        handles=threshold_handles, title="Policy-Threshold Status", loc="upper left",
        bbox_to_anchor=(1.01, 0.65), frameon=False, borderaxespad=0,
        fontsize=9.3, title_fontsize=10.2, handlelength=1.2, handletextpad=0.6,
    )

    colourbar_axis = fig.add_axes([0.805, 0.15, 0.028, 0.26])
    scalar = plt.cm.ScalarMappable(cmap="YlOrBr", norm=Normalize(vmin=0, vmax=MAP_VMAX))
    scalar.set_array([])
    colourbar = fig.colorbar(scalar, cax=colourbar_axis)
    ticks = list(range(0, MAP_VMAX + 1, 10))
    colourbar.set_ticks(ticks)
    colourbar.set_ticklabels(["0"] + [f"{tick}%" for tick in ticks[1:]])
    colourbar.outline.set_linewidth(0.7)

    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(OUT_FIGURES / f"figure_4.{extension}", bbox_inches="tight", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    make_map()
    print(f"wrote {OUT_FIGURES/'figure_4.pdf'}")
