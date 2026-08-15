"""Create publication-ready capacity prediction figures for three datasets.

Each figure follows the evidence structure used in the reference capacity-
prediction figure: a full degradation trajectory with the proposed method and
its absolute error, plus an EOL-region enlargement comparing all methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "figures" / "Models_predicts"
OUTPUT_ROOT = ROOT / "figures" / "capacity_prediction_curves"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    battery: str
    window_start: int
    prediction_start: int
    threshold_ah: float
    raw_path: Path
    raw_cycle_col: str = "Cycle"
    raw_capacity_col: str = "Capacity"
    zoom_half_width: int = 40


SPECS = {
    "NASA": DatasetSpec(
        label="NASA",
        battery="B0005",
        window_start=16,
        prediction_start=50,
        threshold_ah=1.40,
        raw_path=ROOT / "data" / "version3" / "NASA data" / "NASA_Data_minimal_interpolated.npy",
        zoom_half_width=20,
    ),
    "CALCE": DatasetSpec(
        label="CALCE",
        battery="CS2_35",
        window_start=64,
        prediction_start=200,
        threshold_ah=0.77,
        raw_path=ROOT / "data" / "version3" / "CALCE data" / "CALCE_Data.npy",
        zoom_half_width=45,
    ),
    "TJU": DatasetSpec(
        label="TJU",
        battery="CY25-1",
        window_start=64,
        prediction_start=200,
        threshold_ah=1.75,
        raw_path=ROOT / "data" / "version3" / "TJU data" / "TJU_Data_version2_model_adapter.npy",
        zoom_half_width=40,
    ),
}


# The display names below follow the capitalization used by the source papers.
METHOD_FILES = {
    "MGI-DSSM": "MGI-DSSM",
    "MSTEA-Net": "MSTEA-Net",
    "IC2ML": "IC2ML",
    "BATTER-MoE": "Batter-MOE",
    "RUL-Mamba": "RUL-Mamba",
    "PatchFormer": "PathFormer",
    "SG-DiTs": "SG-Dits",
    "Autoformer": "AutoFormer",
    "iTransformer": "iTransformer",
}


COLORS = {
    "MGI-DSSM": "#D1495B",
    "MSTEA-Net": "#3A6EA5",
    "IC2ML": "#6A4C93",
    "BATTER-MoE": "#F28E2B",
    "RUL-Mamba": "#2A9D8F",
    "PatchFormer": "#7B8E57",
    "SG-DiTs": "#C49A00",
    "Autoformer": "#8C8C8C",
    "iTransformer": "#4C78A8",
}


LINESTYLES = {
    "MGI-DSSM": "-",
    "MSTEA-Net": "--",
    "IC2ML": "-.",
    "BATTER-MoE": "-",
    "RUL-Mamba": "--",
    "PatchFormer": "-.",
    "SG-DiTs": ":",
    "Autoformer": "--",
    "iTransformer": ":",
}


def load_raw_capacity(spec: DatasetSpec) -> pd.DataFrame:
    container = np.load(spec.raw_path, allow_pickle=True).reshape(-1)[0]
    raw = container[spec.battery]
    frame = pd.DataFrame(
        {
            "cycle": raw[spec.raw_cycle_col].astype(int),
            "capacity_true_ah": raw[spec.raw_capacity_col].astype(float),
        }
    )
    frame = frame.loc[frame["cycle"] >= spec.window_start].copy()
    frame = frame.sort_values("cycle").drop_duplicates("cycle", keep="first")
    if frame.empty or int(frame["cycle"].iloc[0]) != spec.window_start:
        raise ValueError(f"{spec.label}: real-capacity curve does not start at cycle {spec.window_start}")
    return frame.reset_index(drop=True)


def load_prediction(method: str, spec: DatasetSpec) -> pd.DataFrame:
    directory = METHOD_FILES[method]
    path = INPUT_ROOT / directory / f"{spec.label}_predictions.csv"
    frame = pd.read_csv(path)

    if "battery" not in frame.columns:
        raise ValueError(f"{path}: missing battery column")
    frame = frame.loc[frame["battery"].astype(str) == spec.battery].copy()
    if "start_point" in frame.columns:
        frame = frame.loc[frame["start_point"].astype(int) == spec.prediction_start].copy()

    pred_col = next(
        (name for name in ("capacity_pred", "capacity_pred_ah", "pred_ah") if name in frame.columns),
        None,
    )
    if pred_col is None:
        raise ValueError(f"{path}: no recognized prediction column")

    frame = frame.loc[frame["cycle"].astype(int) >= spec.prediction_start, ["cycle", pred_col]].copy()
    frame["cycle"] = frame["cycle"].astype(int)
    frame[pred_col] = frame[pred_col].astype(float)
    frame = frame.sort_values("cycle").drop_duplicates("cycle", keep="first")
    frame = frame.rename(columns={pred_col: method})

    if frame.empty or int(frame["cycle"].iloc[0]) != spec.prediction_start:
        first = None if frame.empty else int(frame["cycle"].iloc[0])
        raise ValueError(
            f"{spec.label}/{method}: prediction starts at {first}, expected {spec.prediction_start}"
        )
    if not np.isfinite(frame[method]).all():
        raise ValueError(f"{spec.label}/{method}: prediction contains non-finite values")
    return frame.reset_index(drop=True)


def build_source_data(spec: DatasetSpec) -> pd.DataFrame:
    source = load_raw_capacity(spec)
    for method in METHOD_FILES:
        source = source.merge(load_prediction(method, spec), on="cycle", how="left", validate="one_to_one")

    pre_prediction = source["cycle"] < spec.prediction_start
    if source.loc[pre_prediction, list(METHOD_FILES)].notna().any().any():
        raise ValueError(f"{spec.label}: a prediction appears before the declared prediction start")
    for method in METHOD_FILES:
        first_valid = int(source.loc[source[method].notna(), "cycle"].iloc[0])
        if first_valid != spec.prediction_start:
            raise ValueError(f"{spec.label}/{method}: merged curve starts at {first_valid}")

    source["MGI-DSSM absolute error (Ah)"] = (
        source["MGI-DSSM"] - source["capacity_true_ah"]
    ).abs()
    source["EOL threshold (Ah)"] = spec.threshold_ah
    return source


def first_crossing_cycle(cycles: pd.Series, values: pd.Series, threshold: float) -> int:
    valid = values.notna() & (values <= threshold)
    if not valid.any():
        raise ValueError("No EOL threshold crossing found")
    return int(cycles.loc[valid].iloc[0])


def padded_limits(values: np.ndarray, fraction: float = 0.055) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-6)
    return lo - fraction * span, hi + fraction * span


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55, alpha=0.65, dashes=(2.5, 2.5))
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def plot_dataset(spec: DatasetSpec) -> dict[str, float | int | str]:
    source = build_source_data(spec)
    true_eol = first_crossing_cycle(source["cycle"], source["capacity_true_ah"], spec.threshold_ah)
    proposed_eol = first_crossing_cycle(source["cycle"], source["MGI-DSSM"], spec.threshold_ah)

    zoom_lo = max(spec.prediction_start, true_eol - spec.zoom_half_width)
    zoom_hi = min(int(source["cycle"].max()), true_eol + spec.zoom_half_width)
    zoom = source.loc[source["cycle"].between(zoom_lo, zoom_hi)].copy()
    if zoom[list(METHOD_FILES)].isna().any().any():
        raise ValueError(f"{spec.label}: EOL zoom contains missing model predictions")

    zoom_values = [zoom["capacity_true_ah"].to_numpy(), np.array([spec.threshold_ah])]
    zoom_values.extend(zoom[method].to_numpy() for method in METHOD_FILES)
    zoom_ymin, zoom_ymax = padded_limits(np.concatenate(zoom_values), fraction=0.065)

    full_values = np.concatenate(
        [
            source["capacity_true_ah"].to_numpy(),
            source.loc[source["cycle"] >= spec.prediction_start, "MGI-DSSM"].to_numpy(),
            np.array([spec.threshold_ah]),
        ]
    )
    full_ymin, full_ymax = padded_limits(full_values, fraction=0.045)

    fig = plt.figure(figsize=(7.20, 3.55), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.62, 1.0],
        left=0.082,
        right=0.985,
        bottom=0.175,
        top=0.765,
        wspace=0.37,
    )
    ax_full = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_error = ax_full.twinx()

    # Observation/history region and the explicitly requested prediction start.
    ax_full.axvspan(
        spec.window_start,
        spec.prediction_start,
        facecolor="#ECECEC",
        alpha=0.62,
        linewidth=0,
        zorder=0,
    )
    ax_full.axvline(
        spec.prediction_start,
        color="#686868",
        linestyle=(0, (3, 2)),
        linewidth=0.85,
        zorder=2,
    )

    ax_full.plot(
        source["cycle"],
        source["capacity_true_ah"],
        color="#1A1A1A",
        linewidth=1.35,
        label="Measured capacity",
        zorder=5,
    )
    ax_full.plot(
        source["cycle"],
        source["MGI-DSSM"],
        color=COLORS["MGI-DSSM"],
        linewidth=1.65,
        label="MGI-DSSM",
        zorder=6,
    )
    ax_full.axhline(
        spec.threshold_ah,
        color="#3155C6",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )
    ax_error.plot(
        source["cycle"],
        source["MGI-DSSM absolute error (Ah)"],
        color="#2B8C3E",
        linewidth=0.72,
        alpha=0.92,
        zorder=3,
    )

    error_max = float(source["MGI-DSSM absolute error (Ah)"].max())
    ax_error.set_ylim(0, max(0.005, error_max * 1.12))
    ax_error.set_ylabel("")
    ax_error.tick_params(axis="y", colors="#2B8C3E", width=0.7, labelsize=7.1)
    ax_error.spines["right"].set_visible(True)
    ax_error.spines["right"].set_color("#2B8C3E")
    ax_error.spines["right"].set_linewidth(0.75)
    ax_error.spines["top"].set_visible(False)
    ax_error.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Local EOL comparison. Draw quieter baselines first and the key evidence last.
    for method in reversed(list(METHOD_FILES)):
        linewidth = 1.65 if method == "MGI-DSSM" else 0.95
        alpha = 1.0 if method == "MGI-DSSM" else 0.90
        zorder = 8 if method == "MGI-DSSM" else 3
        ax_zoom.plot(
            zoom["cycle"],
            zoom[method],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    ax_zoom.plot(
        zoom["cycle"],
        zoom["capacity_true_ah"],
        color="#1A1A1A",
        linewidth=1.4,
        zorder=9,
    )
    ax_zoom.axhline(
        spec.threshold_ah,
        color="#3155C6",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )
    ax_zoom.axvline(
        true_eol,
        color="#1A1A1A",
        linestyle=(0, (1.5, 2.0)),
        linewidth=0.9,
        zorder=2,
    )
    ax_zoom.axvline(
        proposed_eol,
        color=COLORS["MGI-DSSM"],
        linestyle=(0, (4, 2, 1, 2)),
        linewidth=1.0,
        zorder=2,
    )

    ax_full.set_xlim(spec.window_start, int(source["cycle"].max()))
    ax_full.set_ylim(full_ymin, full_ymax)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_ylim(zoom_ymin, zoom_ymax)
    ax_full.set_xlabel("Cycle")
    ax_full.set_ylabel("Capacity (Ah)")
    ax_zoom.set_xlabel("Cycle")
    ax_zoom.set_ylabel("Capacity (Ah)")
    style_axis(ax_full)
    style_axis(ax_zoom)

    ax_full.set_title(f"Full trajectory ({spec.battery})", pad=6, fontweight="semibold")
    ax_zoom.set_title("EOL-region comparison", pad=6, fontweight="semibold")
    ax_full.text(-0.14, 1.08, "a", transform=ax_full.transAxes, fontsize=10.5, fontweight="bold", va="top")
    ax_zoom.text(-0.20, 1.08, "b", transform=ax_zoom.transAxes, fontsize=10.5, fontweight="bold", va="top")

    history_mid = 0.5 * (spec.window_start + spec.prediction_start)
    ax_full.text(
        history_mid,
        full_ymax - 0.035 * (full_ymax - full_ymin),
        "Observed history",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    ax_full.annotate(
        f"Prediction start: cycle {spec.prediction_start}",
        xy=(spec.prediction_start, full_ymax - 0.11 * (full_ymax - full_ymin)),
        xytext=(7, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#4A4A4A",
    )
    ax_full.text(
        int(source["cycle"].max()),
        spec.threshold_ah + 0.018 * (full_ymax - full_ymin),
        f"EOL threshold = {spec.threshold_ah:.2f} Ah",
        color="#3155C6",
        fontsize=6.8,
        ha="right",
        va="bottom",
    )

    ax_full.text(
        0.985,
        0.975,
        "Absolute error (right axis)",
        transform=ax_full.transAxes,
        color="#2B8C3E",
        fontsize=6.5,
        ha="right",
        va="top",
    )
    eol_box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82)
    ax_zoom.text(
        0.975,
        0.965,
        f"Measured EOL: cycle {true_eol}",
        transform=ax_zoom.transAxes,
        color="#1A1A1A",
        fontsize=6.3,
        ha="right",
        va="top",
        bbox=eol_box,
        zorder=12,
    )
    ax_zoom.text(
        0.975,
        0.895,
        f"MGI-DSSM EOL: cycle {proposed_eol}",
        transform=ax_zoom.transAxes,
        color=COLORS["MGI-DSSM"],
        fontsize=6.3,
        ha="right",
        va="top",
        bbox=eol_box,
        zorder=12,
    )

    # Explicit enlargement box and connectors, mirroring the reference figure's logic.
    zoom_box = Rectangle(
        (zoom_lo, zoom_ymin),
        zoom_hi - zoom_lo,
        zoom_ymax - zoom_ymin,
        facecolor="#BFC7D5",
        edgecolor="#697386",
        linewidth=0.8,
        alpha=0.20,
        zorder=2,
    )
    ax_full.add_patch(zoom_box)
    for y_full, y_zoom in ((zoom_ymin, zoom_ymin), (zoom_ymax, zoom_ymax)):
        connector = ConnectionPatch(
            xyA=(zoom_hi, y_full),
            coordsA=ax_full.transData,
            xyB=(zoom_lo, y_zoom),
            coordsB=ax_zoom.transData,
            color="#697386",
            linewidth=0.65,
            alpha=0.75,
            zorder=1,
            clip_on=False,
        )
        fig.add_artist(connector)

    legend_handles = [Line2D([0], [0], color="#1A1A1A", lw=1.5, label="Measured capacity")]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            lw=1.7 if method == "MGI-DSSM" else 1.05,
            linestyle=LINESTYLES[method],
            label=method,
        )
        for method in METHOD_FILES
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.965),
        ncol=5,
        columnspacing=1.35,
        handlelength=2.6,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    fig.text(
        0.535,
        0.995,
        f"{spec.label} dataset capacity prediction",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_ROOT / f"capacity_prediction_{spec.label}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    # Submission portals commonly expect composite line art as RGB rather than RGBA.
    with Image.open(tiff_path) as rendered_tiff:
        rendered_tiff.convert("RGB").save(tiff_path, compression="tiff_lzw", dpi=(600, 600))

    source_path = OUTPUT_ROOT / f"source_data_{spec.label}.csv"
    source.to_csv(source_path, index=False, float_format="%.10g")
    return {
        "dataset": spec.label,
        "battery": spec.battery,
        "real_start": int(source["cycle"].min()),
        "prediction_start": spec.prediction_start,
        "end_cycle": int(source["cycle"].max()),
        "true_eol": true_eol,
        "proposed_eol": proposed_eol,
        "zoom_start": zoom_lo,
        "zoom_end": zoom_hi,
        "rows": int(len(source)),
    }


def main() -> None:
    summaries = [plot_dataset(SPECS[name]) for name in ("NASA", "CALCE", "TJU")]
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
