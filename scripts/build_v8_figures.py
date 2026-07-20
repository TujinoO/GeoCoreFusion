"""Build the publication-ready V8 research figure set.

The script reads the frozen V8 evidence tables and report JSON.  Figure 04
reconstructs only the four declared 3DSSZ Wald views using the existing
benchmark implementation; pseudo-HR HSI truth is used for evaluation/display
only and is never passed into candidate estimation.

Every figure is exported as a 300-dpi PNG and a vector PDF.  The manifest
records concise alternative text, the defensible claim scope, and the source
evidence for each figure.  No experiment artifact or core implementation is
modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import (
    Arc,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from PIL import Image


NAVY = "#16324F"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#8C6BB1"
GRAY = "#6B7280"
DARK_GRAY = "#374151"
MID_GRAY = "#9CA3AF"
LIGHT = "#F5F7FA"
PALE_BLUE = "#E5F1F8"
PALE_GREEN = "#E5F5EF"
PALE_ORANGE = "#FFF1D6"
PALE_RED = "#FCE8E2"
GRID = "#D8DEE8"
WHITE = "#FFFFFF"

SCENE_COLORS = {"3dssz": BLUE, "zkh3": ORANGE}
SCENE_LABELS = {"3dssz": "3DSSZ", "zkh3": "ZKH3"}
STATUS_COLORS = {
    "unidentifiable": GRAY,
    "weakly_identifiable": ORANGE,
    "identifiable": GREEN,
}

FIGURE_STEMS = [
    "01_v8_conditional_route",
    "02_identifiability_spectral_map",
    "03_wald_cross_scene_methods",
    "04_wald_2201_visual_truth",
    "05_spectral_cone_geometry",
    "06_3dssz_pareto_failure",
    "07_registration_uncertainty_sigma_points",
    "08_scientific_vs_visualization_contract",
    "09_validation_counterfactual_matrix",
    "10_truth_acquisition_roadmap",
]


@dataclass(slots=True)
class Evidence:
    identifiability: pd.DataFrame
    wald_summary: pd.DataFrame
    wald_bands: pd.DataFrame
    detail_sweep: pd.DataFrame
    wald_report: dict[str, Any]
    method_decision_text: str
    source_paths: dict[str, Path]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": DARK_GRAY,
            "axes.edgecolor": GRID,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_evidence(repo: Path) -> Evidence:
    root = repo / "artifacts" / "v7_research" / "evidence"
    paths = {
        "identifiability": root / "v8_identifiability.csv",
        "wald_summary": root / "v8_wald_summary.csv",
        "wald_bands": root / "v8_wald_band_metrics.csv",
        "detail_sweep": root / "v8_detail_sweep.csv",
        "wald_report": root / "v8_wald_benchmark.json",
        "method_decision": root / "v8_method_decision.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V8 evidence: " + ", ".join(missing))
    return Evidence(
        identifiability=pd.read_csv(paths["identifiability"], encoding="utf-8-sig"),
        wald_summary=pd.read_csv(paths["wald_summary"], encoding="utf-8-sig"),
        wald_bands=pd.read_csv(paths["wald_bands"], encoding="utf-8-sig"),
        detail_sweep=pd.read_csv(paths["detail_sweep"], encoding="utf-8-sig"),
        wald_report=json.loads(paths["wald_report"].read_text(encoding="utf-8")),
        method_decision_text=paths["method_decision"].read_text(encoding="utf-8"),
        source_paths=paths,
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    face: str = PALE_BLUE,
    edge: str = BLUE,
    title_color: str = NAVY,
    body_color: str = DARK_GRAY,
    title_size: float = 10.3,
    body_size: float = 8.5,
    linewidth: float = 1.5,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.69,
        title,
        ha="center",
        va="center",
        color=title_color,
        weight="bold",
        fontsize=title_size,
        transform=ax.transAxes,
    )
    ax.text(
        x + width / 2,
        y + height * 0.32,
        body,
        ha="center",
        va="center",
        color=body_color,
        fontsize=body_size,
        linespacing=1.35,
        transform=ax.transAxes,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    width: float = 1.5,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            shrinkA=3,
            shrinkB=3,
            connectionstyle=connectionstyle,
        )
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    alt: str,
    claim_scope: str,
    sources: list[str],
) -> dict[str, Any]:
    if stem not in FIGURE_STEMS:
        raise ValueError(f"Unexpected V8 figure stem: {stem}")
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    with Image.open(png_path) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (300.0, 300.0))
    return {
        "stem": stem,
        "png": png_path.name,
        "pdf": pdf_path.name,
        "pixel_size": [int(width), int(height)],
        "png_dpi": [round(float(dpi[0]), 2), round(float(dpi[1]), 2)],
        "alt": alt,
        "claim_scope": claim_scope,
        "sources": sources,
    }


def _figure_01_route_wide(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    policy = evidence.wald_report["fail_closed_policy"]
    decisions = policy["decisions"]
    angle = float(evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"])
    scene_count = len(decisions)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "Conditional UARF-Cycle：先回答“能否迁移”，再回答“如何受约束迁移”",
        transform=ax.transAxes,
        fontsize=19,
        weight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.02,
        0.918,
        "候选无关的可辨识性门控决定分支；不可辨识时不以更大 gain 强行补细节",
        transform=ax.transAxes,
        fontsize=11,
        color=DARK_GRAY,
        va="top",
    )

    add_box(
        ax,
        (0.02, 0.67),
        0.14,
        0.16,
        "HR-RGB",
        "几何参考与候选特征\n不直接定义 SWIR 真值",
        face=PALE_BLUE,
        edge=BLUE,
    )
    add_box(
        ax,
        (0.02, 0.43),
        0.14,
        0.16,
        "原生 NIR / SWIR",
        "原生 detector bands\n不把插值波段当独立观测",
        face=PALE_GREEN,
        edge=GREEN,
    )
    add_box(
        ax,
        (0.02, 0.19),
        0.14,
        0.16,
        "标定与不确定性",
        "band-wise MTF / dark-flat\n注册协方差与有效像元",
        face=LIGHT,
        edge=GRAY,
    )

    add_box(
        ax,
        (0.205, 0.62),
        0.19,
        0.22,
        "同 MTF 的 log-DoG 通带",
        "RGB 与原生波段进入同一可观测通带\n空间留块 ridge + shift-null\n门控不读取任何融合候选",
        face=PALE_BLUE,
        edge=BLUE,
        body_size=8.3,
    )
    add_box(
        ax,
        (0.205, 0.30),
        0.19,
        0.20,
        "Observation-only 基线 X0",
        "只由原生观测、退化算子与\n冻结正则得到；始终可回退",
        face=PALE_GREEN,
        edge=GREEN,
    )
    add_arrow(ax, (0.16, 0.75), (0.205, 0.73), color=BLUE)
    add_arrow(ax, (0.16, 0.51), (0.205, 0.69), color=GREEN)
    add_arrow(ax, (0.16, 0.27), (0.205, 0.40), color=GRAY)

    add_box(
        ax,
        (0.44, 0.52),
        0.17,
        0.22,
        "Band / region gate",
        "identifiable？\nweak / unidentifiable /\ninsufficient support 均 fail-closed",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        body_size=8.5,
    )
    add_arrow(ax, (0.395, 0.73), (0.44, 0.65), color=BLUE)
    add_arrow(ax, (0.395, 0.40), (0.44, 0.58), color=GREEN)

    add_box(
        ax,
        (0.66, 0.65),
        0.18,
        0.20,
        "Gate ON：受约束候选",
        f"rank-8 coefficient ridge\n固定 {angle:.1f}° spectral cone\n逐原生 band forward refinement",
        face=PALE_BLUE,
        edge=BLUE,
        body_size=8.4,
    )
    add_box(
        ax,
        (0.66, 0.35),
        0.18,
        0.18,
        "Gate OFF：明确不恢复",
        "输出 X0；把不可辨识高频\n保留为“不知道”，不复制 RGB",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=8.5,
    )
    add_arrow(ax, (0.61, 0.65), (0.66, 0.75), color=BLUE)
    add_arrow(ax, (0.61, 0.58), (0.66, 0.44), color=VERMILION)

    add_box(
        ax,
        (0.88, 0.65),
        0.105,
        0.20,
        "Scientific cube",
        "可用于光谱与矿物分析\n附 gate / cone / forward QA",
        face=PALE_GREEN,
        edge=GREEN,
        title_size=9.5,
        body_size=7.7,
    )
    add_box(
        ax,
        (0.88, 0.35),
        0.105,
        0.18,
        "Observation-only",
        "科学输出的安全回退\n不声称补回 SWIR 高频",
        face=LIGHT,
        edge=GRAY,
        title_size=9.3,
        body_size=7.7,
    )
    add_arrow(ax, (0.84, 0.75), (0.88, 0.75), color=GREEN)
    add_arrow(ax, (0.84, 0.44), (0.88, 0.44), color=GRAY)

    add_box(
        ax,
        (0.44, 0.12),
        0.40,
        0.14,
        "独立的 RGB-textured visualization",
        "只服务肉眼判读；独立目录、manifest 与显式警示；不得进入定量光谱",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        body_size=8.4,
    )
    add_arrow(
        ax,
        (0.75, 0.35),
        (0.73, 0.26),
        color=ORANGE,
        connectionstyle="arc3,rad=0.18",
    )

    gate_3dssz = str(decisions["3dssz"]["gate_state"]).upper()
    gate_zkh3 = str(decisions["zkh3"]["gate_state"]).upper()
    ax.text(
        0.02,
        0.075,
        f"当前证据：3DSSZ gate {gate_3dssz}；ZKH3 gate {gate_zkh3} 仅为 conditional proof-of-strategy。",
        transform=ax.transAxes,
        fontsize=10.3,
        weight="bold",
        color=VERMILION,
    )
    ax.text(
        0.02,
        0.035,
        f"声明边界：仅 {scene_count} 个场景，尚无独立 HR-NIR/SWIR 真值；该路线是冻结研究合同，不是 V8 已发布或“SWIR 细节无损”结论。",
        transform=ax.transAxes,
        fontsize=9.6,
        color=DARK_GRAY,
    )
    return save_figure(
        fig,
        output_dir,
        "01_v8_conditional_route",
        alt=(
            "Conditional UARF-Cycle 总路线：RGB、原生 NIR/SWIR 与标定信息进入候选无关的 band-pass gate；"
            "可辨识分支使用 rank-8、固定光谱锥和原生 forward，不可辨识分支回退 observation-only。"
        ),
        claim_scope=(
            "冻结的下一轮研究合同；3DSSZ 关闭、ZKH3 条件开启只与当前两个场景一致，"
            "不构成 V8 发布或真实 SWIR 高频无损恢复声明。"
        ),
        sources=["v8_method_decision.md", "v8_wald_benchmark.json"],
    )


def figure_01_route(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    policy = evidence.wald_report["fail_closed_policy"]
    decisions = policy["decisions"]
    angle = float(evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"])
    ident = evidence.identifiability
    ident = ident[(ident["model"] == "rgb3_ridge") & (ident["scope"] == "all_valid")]
    scene_counts: dict[str, dict[str, int]] = {}
    for scene in ("3DSSZ", "ZKH3"):
        counts = ident[ident["scene"] == scene]["identifiability"].value_counts()
        scene_counts[scene] = {
            "I": int(counts.get("identifiable", 0)),
            "W": int(counts.get("weakly_identifiable", 0)),
            "U": int(counts.get("unidentifiable", 0)),
            "total": int(counts.sum()),
        }
    scene_count = len(decisions)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    ax.axis("off")
    ax.text(
        0.02,
        0.975,
        "Conditional UARF-Cycle：先判断“能否迁移”，再执行受约束迁移",
        transform=ax.transAxes,
        fontsize=15.2,
        weight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.02,
        0.932,
        "候选无关 gate 控制 scientific 分支；不可辨识时不以更大 gain 强行补细节",
        transform=ax.transAxes,
        fontsize=8.5,
        color=DARK_GRAY,
        va="top",
    )

    input_boxes = [
        (0.03, "HR-RGB", "几何参考 + 候选特征\n不定义 SWIR 真值", PALE_BLUE, BLUE),
        (0.36, "原生 NIR / SWIR", "detector bands\n不把插值行当独立观测", PALE_GREEN, GREEN),
        (0.69, "标定 / 不确定性", "band-wise MTF / noise\n注册 covariance / mask", LIGHT, GRAY),
    ]
    for x, title, body, face, edge in input_boxes:
        add_box(
            ax,
            (x, 0.77),
            0.28,
            0.12,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
            title_size=9.0,
            body_size=6.8,
        )

    add_box(
        ax,
        (0.05, 0.59),
        0.42,
        0.12,
        "候选无关的同 MTF band-pass gate",
        "RGB / HSI 进入同一 log-DoG 通带\n空间留块 ridge + RGB shift-null",
        face=PALE_BLUE,
        edge=BLUE,
        title_size=9.0,
        body_size=6.8,
    )
    add_box(
        ax,
        (0.53, 0.59),
        0.42,
        0.12,
        "Observation-only 基线 X0",
        "只由原生观测、退化算子与冻结正则得到\n所有区域始终允许安全回退",
        face=PALE_GREEN,
        edge=GREEN,
        title_size=9.0,
        body_size=6.8,
    )
    add_arrow(ax, (0.17, 0.77), (0.20, 0.71), color=BLUE)
    add_arrow(ax, (0.50, 0.77), (0.39, 0.71), color=GREEN)
    add_arrow(ax, (0.83, 0.77), (0.78, 0.71), color=GRAY)

    add_box(
        ax,
        (0.25, 0.43),
        0.50,
        0.10,
        "Band / region 决策",
        "identifiable → ON；weak / unidentifiable / insufficient support → OFF",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        title_size=9.2,
        body_size=6.9,
    )
    add_arrow(ax, (0.29, 0.59), (0.40, 0.53), color=BLUE)
    add_arrow(ax, (0.71, 0.59), (0.60, 0.53), color=GREEN)

    counts_3 = scene_counts["3DSSZ"]
    counts_z = scene_counts["ZKH3"]
    add_box(
        ax,
        (0.04, 0.24),
        0.43,
        0.13,
        f"3DSSZ｜gate {str(decisions['3dssz']['gate_state']).upper()}｜scientific X0",
        f"I/W/U = {counts_3['I']}/{counts_3['W']}/{counts_3['U']}（共 {counts_3['total']}）\n"
        "明确不恢复不可辨识 RGB 高频",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        title_size=8.6,
        body_size=6.8,
    )
    add_box(
        ax,
        (0.53, 0.24),
        0.43,
        0.13,
        "ZKH3｜CONDITIONAL｜受约束 scientific cube",
        f"I/W/U = {counts_z['I']}/{counts_z['W']}/{counts_z['U']}（共 {counts_z['total']}）\n"
        f"rank-8 ridge + 固定 {angle:.1f}° cone + native forward",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        title_size=8.4,
        body_size=6.7,
    )
    add_arrow(ax, (0.40, 0.43), (0.28, 0.37), color=VERMILION)
    add_arrow(ax, (0.60, 0.43), (0.72, 0.37), color=GREEN)

    add_box(
        ax,
        (0.26, 0.095),
        0.48,
        0.085,
        "独立 RGB-textured visualization",
        "可追求接近 RGB 的观感；独立目录 / manifest；不得用于定量光谱",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        title_size=8.8,
        body_size=6.6,
    )
    add_arrow(ax, (0.74, 0.24), (0.62, 0.18), color=ORANGE, connectionstyle="arc3,rad=-0.12")
    ax.text(
        0.02,
        0.040,
        f"声明边界：仅 {scene_count} 个场景；3DSSZ OFF、ZKH3 conditional 不是分类器验证。尚无独立 HR-NIR/SWIR 真值。",
        transform=ax.transAxes,
        fontsize=8.0,
        weight="bold",
        color=VERMILION,
    )
    return save_figure(
        fig,
        output_dir,
        "01_v8_conditional_route",
        alt=(
            "Conditional UARF-Cycle 总路线：RGB、原生 NIR/SWIR 与标定信息进入候选无关的 band-pass gate；"
            "3DSSZ 关闭并输出 observation-only，ZKH3 仅条件进入 rank-8、固定光谱锥和原生 forward 分支。"
        ),
        claim_scope=(
            "冻结的下一轮研究合同；3DSSZ 关闭、ZKH3 条件开启只与当前两个场景一致，"
            "不构成 V8 发布或真实 SWIR 高频无损恢复声明。"
        ),
        sources=["v8_method_decision.md", "v8_identifiability.csv", "v8_wald_benchmark.json"],
    )


def _representative_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for wavelength in (901.0, 1651.0, 2201.0, 2351.0):
        index = (frame["wavelength_nm"] - wavelength).abs().idxmin()
        rows.append(frame.loc[index])
    return pd.DataFrame(rows)


def figure_02_identifiability(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    data = evidence.identifiability
    selected = data[(data["model"] == "rgb3_ridge") & (data["scope"] == "all_valid")]
    scenes = ["3DSSZ", "ZKH3"]
    fig = plt.figure(figsize=(9.0, 6.5))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.6, 1.35),
        height_ratios=(1, 1),
        hspace=0.30,
        wspace=0.15,
    )
    for row_index, scene in enumerate(scenes):
        frame = selected[selected["scene"] == scene].sort_values("wavelength_nm")
        ax = fig.add_subplot(grid[row_index, 0])
        x = frame["wavelength_nm"].to_numpy(dtype=float)
        r2 = frame["cv_pooled_r2"].to_numpy(dtype=float)
        status = frame["identifiability"].astype(str).to_numpy()
        ax.axhline(0.0, color=GRID, lw=0.9, zorder=0)
        ax.plot(x, r2, color=NAVY, lw=1.1, alpha=0.75, zorder=1)
        for label in ("unidentifiable", "weakly_identifiable", "identifiable"):
            mask = status == label
            ax.scatter(
                x[mask],
                r2[mask],
                s=14,
                color=STATUS_COLORS[label],
                edgecolors="none",
                label={
                    "unidentifiable": "unidentifiable",
                    "weakly_identifiable": "weak",
                    "identifiable": "identifiable",
                }[label],
                zorder=2,
            )
        reps = _representative_rows(frame)
        for _, record in reps.iterrows():
            wave = float(record["wavelength_nm"])
            value = float(record["cv_pooled_r2"])
            if wave >= 2300.0:
                offset = -18
            elif wave >= 2150.0:
                offset = 13
            else:
                offset = 11 if value <= 0.60 else -18
            ax.annotate(
                f"{wave:.0f} nm  R²={value:.3f}",
                (wave, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.0,
                color=DARK_GRAY,
                arrowprops=dict(arrowstyle="-", color=MID_GRAY, lw=0.7),
            )
        ax.set_xlim(float(x.min()) - 15.0, float(x.max()) + 15.0)
        y_min = min(-0.10, float(np.nanmin(r2)) - 0.04)
        y_max = max(0.72, float(np.nanmax(r2)) + 0.08)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("空间留块 predictive R²")
        ax.set_xlabel("谐调输出波长（nm；相邻 5 nm 行不视为独立重复）")
        ax.grid(axis="y", color=GRID, lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        gate = "OFF" if scene == "3DSSZ" else "CONDITIONAL"
        gate_color = VERMILION if scene == "3DSSZ" else ORANGE
        ax.set_title(
            f"{scene}｜全有效区域｜rgb3 ridge｜scientific gate {gate}",
            loc="left",
            color=gate_color,
            fontsize=13,
        )
        if row_index == 0:
            ax.legend(frameon=False, ncol=3, loc="upper left")

        count_ax = fig.add_subplot(grid[row_index, 1])
        count_ax.axis("off")
        counts = frame["identifiability"].value_counts().to_dict()
        total = len(frame)
        y_positions = [0.70, 0.48, 0.26]
        for y, label, short in zip(
            y_positions,
            ("identifiable", "weakly_identifiable", "unidentifiable"),
            ("I", "W", "U"),
            strict=True,
        ):
            value = int(counts.get(label, 0))
            count_ax.add_patch(
                FancyBboxPatch(
                    (0.08, y),
                    0.84,
                    0.15,
                    boxstyle="round,pad=0.012,rounding_size=0.02",
                    facecolor={
                        "identifiable": PALE_GREEN,
                        "weakly_identifiable": PALE_ORANGE,
                        "unidentifiable": LIGHT,
                    }[label],
                    edgecolor=STATUS_COLORS[label],
                    linewidth=1.3,
                    transform=count_ax.transAxes,
                )
            )
            count_ax.text(
                0.18,
                y + 0.075,
                short,
                transform=count_ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                weight="bold",
                color=DARK_GRAY,
            )
            count_ax.text(
                0.86,
                y + 0.075,
                f"{value}/{total}",
                transform=count_ax.transAxes,
                ha="right",
                va="center",
                fontsize=13,
                weight="bold",
                color=STATUS_COLORS[label],
            )
        count_ax.text(
            0.08,
            0.91,
            "全谱状态计数",
            transform=count_ax.transAxes,
            fontsize=11.3,
            weight="bold",
            color=NAVY,
        )
        count_ax.text(
            0.08,
            0.08,
            "状态来自候选无关的\n同 MTF band-pass 留块验证",
            transform=count_ax.transAxes,
            fontsize=8.5,
            color=DARK_GRAY,
            linespacing=1.4,
        )

    fig.suptitle(
        "全波段可辨识性地图：同一方法在两场景给出相反的注入许可",
        fontsize=17.5,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "限制：当前只有 3DSSZ 与 ZKH3 两场景；ZKH3 的 conditional 状态不是已验证分类器，正式实现还必须回到原生 detector bands。",
        ha="center",
        fontsize=9.5,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(
        fig,
        output_dir,
        "02_identifiability_spectral_map",
        alt=(
            "3DSSZ 与 ZKH3 全波段 predictive R² 和可辨识状态图。3DSSZ 367 个波段均不可辨识，"
            "ZKH3 全有效区为 365 个 identifiable、2 个 weak。"
        ),
        claim_scope=(
            "候选无关 LR 同通带的两场景诊断；支持 3DSSZ 关闭和 ZKH3 条件分支，"
            "不证明 LR Nyquist 以上 RGB 高频等于真实 SWIR 高频。"
        ),
        sources=["v8_identifiability.csv", "v8_identifiability.md"],
    )


def figure_03_wald_methods(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    data = evidence.wald_summary.copy()
    method_order = evidence.wald_report["method_order"]
    non_baseline = [method for method in method_order if method != "bicubic"]
    abbreviations = {
        "mtf_glp_gsa_additive": "MTF-GLP/GSA",
        "mtf_glp_hpm_linear": "linear HPM",
        "mtf_glp_hpm_log": "log HPM",
        "brovey_cn_luma_ratio": "Brovey/CN",
        "rgb_ridge_global_multifeature": "global ridge",
        "rgb_ridge_blocked_cv_multifeature": "blocked-CV ridge",
        "lowrank_r8_rgb_ridge": "rank-8 ridge",
        "lowrank_r8_rgb_ridge_cone05": "rank-8 + cone05",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.8), sharey=True)
    panels = [
        ("rmse_improvement_vs_bicubic_pct", "全谱 RMSE 收益 vs bicubic"),
        ("2201nm_rmse_improvement_vs_bicubic_pct", "2201 nm RMSE 收益 vs bicubic"),
    ]
    y = np.arange(len(non_baseline), dtype=float)
    offset = 0.18
    for ax, (metric, title) in zip(axes, panels, strict=True):
        ax.axvline(0.0, color=DARK_GRAY, lw=1.0, zorder=0)
        for scene, delta in (("3dssz", -offset), ("zkh3", offset)):
            frame = data[data["scene"] == scene].set_index("method")
            values = np.asarray([float(frame.loc[method, metric]) for method in non_baseline])
            bars = ax.barh(
                y + delta,
                values,
                height=0.31,
                color=SCENE_COLORS[scene],
                alpha=0.88,
                label=SCENE_LABELS[scene],
                zorder=2,
            )
            for bar, value, method in zip(bars, values, non_baseline, strict=True):
                if method != "lowrank_r8_rgb_ridge_cone05":
                    continue
                x_text = value + (1.0 if value >= 0 else -1.0)
                ax.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.2f}%",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color=SCENE_COLORS[scene],
                )
        ax.set_yticks(y, [abbreviations[method] for method in non_baseline])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE 改善（%；正值更好）")
        ax.set_title(title, color=NAVY, fontsize=12.5)
        ax.grid(axis="x", color=GRID, lw=0.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles[:2],
        legend_labels[:2],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
    )

    policy = evidence.wald_report["fail_closed_policy"]["decisions"]
    selected_3 = policy["3dssz"]["selected_method"]
    selected_z = policy["zkh3"]["selected_method"]
    axes[0].text(
        0.02,
        0.015,
        f"3DSSZ gate OFF → {selected_3}",
        transform=axes[0].transAxes,
        color=VERMILION,
        fontsize=10,
        weight="bold",
        va="bottom",
    )
    axes[1].text(
        0.98,
        0.015,
        f"ZKH3 conditional → {abbreviations[selected_z]}",
        transform=axes[1].transAxes,
        color=ORANGE,
        fontsize=10,
        weight="bold",
        va="bottom",
        ha="right",
    )
    fig.suptitle(
        "严格 Wald 跨场景结果：无条件注入不可防守，cone05 也必须由 gate 控制",
        fontsize=17,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.016,
        "伪 HR 真值仅在候选冻结后用于评价；3DSSZ 与 ZKH3 的收益方向不同。当前两场景一致性不等于独立验证。",
        ha="center",
        fontsize=9.6,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.94), w_pad=2.2)
    return save_figure(
        fig,
        output_dir,
        "03_wald_cross_scene_methods",
        alt=(
            "八种非 bicubic 方法在 3DSSZ 和 ZKH3 的全谱及 2201 nm Wald RMSE 收益对比。"
            "rank-8 加 0.5 度光谱锥在 3DSSZ 为负收益，在 ZKH3 为正收益。"
        ),
        claim_scope=(
            "严格内部 Wald 伪真值排序；支持 fail-closed 条件策略，不替代真实仪器分辨率下的独立 HR-NIR/SWIR 真值。"
        ),
        sources=["v8_wald_summary.csv", "v8_wald_benchmark.json"],
    )


def _load_wald_module(repo: Path) -> Any:
    script = repo / "scripts" / "run_v8_wald_benchmark.py"
    spec = importlib.util.spec_from_file_location("geocorefusion_v8_wald", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Wald benchmark from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reconstruct_3dssz_wald_2201(
    repo: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the four frozen visual panels without truth leakage."""

    module = _load_wald_module(repo)
    protocol = module.WaldProtocol()
    truth, rgb_hr, hsi_lr, wavelengths, scene_meta = module._load_scene(
        repo, "3dssz", protocol
    )
    high_shape = rgb_hr.shape[:2]
    base = module._clip_candidate(module.upsample(hsi_lr, high_shape), hsi_lr)
    feature_hr, feature_names = module.rgb_features(rgb_hr)
    feature_lr = module.degrade(feature_hr, protocol)

    # This block is the benchmark's frozen rank-8 branch, isolated here so the
    # figure builder does not materialize the seven unused full-cube candidates.
    y = hsi_lr.reshape(-1, hsi_lr.shape[2]).astype(np.float64)
    y_mean = np.mean(y, axis=0)
    centered = y - y_mean
    rank = min(8, hsi_lr.shape[2], y.shape[0] - 1)
    _, _, vt_lowrank = np.linalg.svd(centered, full_matrices=False)
    basis = vt_lowrank[:rank].astype(np.float32)
    coefficient_lr = (centered @ basis.T).reshape(hsi_lr.shape[:2] + (rank,))
    coefficient_model, coefficient_cv = module.blocked_cv_ridge(
        feature_lr, coefficient_lr, protocol
    )
    coefficient_prediction_hr = module.predict_ridge(
        coefficient_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape + (rank,))
    coefficient_detail = coefficient_prediction_hr - module.upsample(
        module.degrade(coefficient_prediction_hr, protocol), high_shape
    )
    spectral_detail = np.einsum(
        "...k,kb->...b", coefficient_detail, basis, optimize=True
    ).astype(np.float32)
    spectral_detail = module._clip_detail(
        spectral_detail, hsi_lr, protocol.detail_clip_sigma
    )
    lowrank = module._clip_candidate(base + spectral_detail, hsi_lr)
    cone, diagnostics = module.project_spectral_cone(
        lowrank,
        base,
        protocol.spectral_cone_half_angle_deg,
        return_diagnostics=True,
    )

    index = int(np.argmin(np.abs(wavelengths - 2200.0)))
    images = {
        "truth": np.asarray(truth[:, :, index], dtype=np.float32).copy(),
        "bicubic": np.asarray(base[:, :, index], dtype=np.float32).copy(),
        "lowrank_r8_rgb_ridge": np.asarray(lowrank[:, :, index], dtype=np.float32).copy(),
        "lowrank_r8_rgb_ridge_cone05": np.asarray(cone[:, :, index], dtype=np.float32).copy(),
    }
    metadata = {
        "actual_wavelength_nm": float(wavelengths[index]),
        "rank": int(rank),
        "feature_names": feature_names,
        "coefficient_cv_r2_median": float(coefficient_cv["cv_r2_median"]),
        "cone_clip_fraction_reconstructed": float(
            diagnostics.to_dict()["cone_clip_fraction"]
        ),
        "scene_meta": scene_meta,
    }
    return images, metadata


def _select_texture_crop(image: np.ndarray, margin: int = 8) -> tuple[int, int, int, int]:
    height, width = image.shape
    available = min(height - 2 * margin, width - 2 * margin)
    size = max(40, min(88, int(available)))
    safe = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    local_gradient = cv2.boxFilter(
        gradient,
        -1,
        (size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT101,
    )
    local_mean = cv2.boxFilter(
        safe,
        -1,
        (size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT101,
    )
    valid = safe[margin : height - margin, margin : width - margin]
    median = float(np.nanmedian(valid))
    spread = max(float(np.nanpercentile(valid, 90) - np.nanpercentile(valid, 10)), 1e-6)
    dark_weight = 1.0 + 0.25 * np.clip((median - local_mean) / spread, -1.0, 1.0)
    score = local_gradient * dark_weight
    half = size // 2
    score[: margin + half, :] = -np.inf
    score[height - margin - (size - half) :, :] = -np.inf
    score[:, : margin + half] = -np.inf
    score[:, width - margin - (size - half) :] = -np.inf
    center_y, center_x = np.unravel_index(int(np.nanargmax(score)), score.shape)
    x = int(np.clip(center_x - half, margin, width - margin - size))
    y = int(np.clip(center_y - half, margin, height - margin - size))
    return x, y, size, size


def figure_04_wald_visual(
    repo: Path, evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    images, metadata = _reconstruct_3dssz_wald_2201(repo)
    truth = images["truth"]
    x, y, width, height = _select_texture_crop(truth)
    margin = int(evidence.wald_report["protocol"]["evaluation_margin_hr"])
    valid_truth = truth[margin : truth.shape[0] - margin, margin : truth.shape[1] - margin]
    low, high = np.nanpercentile(valid_truth, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(valid_truth)), float(np.nanmax(valid_truth))

    records = evidence.wald_summary[evidence.wald_summary["scene"] == "3dssz"].set_index(
        "method"
    )
    columns = [
        ("truth", "伪 HR-HSI 真值\n仅评价 / 展示读取"),
        ("bicubic", "Bicubic"),
        ("lowrank_r8_rgb_ridge", "Rank-8 coefficient ridge"),
        ("lowrank_r8_rgb_ridge_cone05", "Rank-8 + cone05"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 7.4))
    for column, (key, label) in enumerate(columns):
        image = images[key]
        axes[0, column].imshow(image, cmap="gray", vmin=low, vmax=high)
        axes[0, column].add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor="none",
                edgecolor=ORANGE,
                linewidth=1.8,
            )
        )
        axes[0, column].set_title(label, color=NAVY, fontsize=11.0)
        axes[0, column].axis("off")
        axes[1, column].imshow(
            image[y : y + height, x : x + width], cmap="gray", vmin=low, vmax=high
        )
        axes[1, column].axis("off")
        if key == "truth":
            subtitle = "局部纹理 ROI\n由真值梯度自动选取"
        else:
            record = records.loc[key]
            subtitle = (
                f"PSNR={float(record['2201nm_psnr_db']):.3f} dB\n"
                f"RMSE={float(record['2201nm_rmse']):.5f}"
            )
        axes[1, column].set_title(
            subtitle,
            fontsize=7.8,
            color=VERMILION if key == "lowrank_r8_rgb_ridge" else DARK_GRAY,
        )
    wave = float(metadata["actual_wavelength_nm"])
    fig.suptitle(
        f"3DSSZ 严格 Wald {wave:.0f} nm：统一显示域下的伪 HR 真值与冻结候选",
        fontsize=16.5,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.032,
        f"四列共享伪 HR 真值 P2–P98 线性显示域 [{low:.4f}, {high:.4f}]；候选估计只读取额外降质 HSI 与伪 HR RGB。",
        ha="center",
        fontsize=9.2,
        color=DARK_GRAY,
    )
    fig.text(
        0.5,
        0.010,
        "此处 truth 是 harmonized_lowres 再降质形成的内部伪真值，不是独立 HR-SWIR；视觉接近或 forward 一致均不能单独证明真实细节。",
        ha="center",
        fontsize=9.3,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.062, 0.99, 0.94), h_pad=1.5, w_pad=0.5)
    return save_figure(
        fig,
        output_dir,
        "04_wald_2201_visual_truth",
        alt=(
            "3DSSZ 2201 nm 严格 Wald 的伪 HR 真值、bicubic、rank-8 ridge 和 rank-8 加光谱锥的全图及局部纹理对比；四列使用同一显示范围。"
        ),
        claim_scope=(
            "内部伪 HR-HSI 降质实验的评价展示；候选无 truth 泄漏，但该 truth 仍不是独立 HR-SWIR，不能支持真实细节无损声明。"
        ),
        sources=["run_v8_wald_benchmark.py", "v8_wald_summary.csv"],
    )


def figure_05_cone(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    angle = float(evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"])
    data = evidence.wald_summary.set_index(["scene", "method"])
    cone_method = "lowrank_r8_rgb_ridge_cone05"
    clip = [float(data.loc[(scene, cone_method), "cone_clip_fraction"]) for scene in ("3dssz", "zkh3")]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.set_xlim(-0.3, 6.3)
    ax.set_ylim(-0.9, 4.0)
    ax.axis("off")
    origin = np.asarray([0.25, 0.0])
    axis_end = np.asarray([5.85, 0.0])
    display_angle = math.radians(18.0)
    upper = origin + np.asarray([5.1, 5.1 * math.tan(display_angle)])
    lower = origin + np.asarray([5.1, -5.1 * math.tan(display_angle)])
    polygon = Polygon(
        [origin, upper, lower],
        closed=True,
        facecolor=PALE_GREEN,
        edgecolor="none",
        alpha=0.85,
    )
    ax.add_patch(polygon)
    ax.plot([origin[0], axis_end[0]], [origin[1], axis_end[1]], color=NAVY, lw=2.0)
    ax.plot([origin[0], upper[0]], [origin[1], upper[1]], color=GREEN, lw=1.3)
    ax.plot([origin[0], lower[0]], [origin[1], lower[1]], color=GREEN, lw=1.3)
    candidate = np.asarray([4.7, 2.85])
    projected = np.asarray([4.7, (4.7 - origin[0]) * math.tan(display_angle)])
    ax.annotate(
        "",
        xy=candidate,
        xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=VERMILION, lw=2.2, mutation_scale=15),
    )
    ax.annotate(
        "",
        xy=projected,
        xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4, mutation_scale=15),
    )
    ax.plot([candidate[0], projected[0]], [candidate[1], projected[1]], color=ORANGE, lw=1.4, ls="--")
    ax.scatter(*origin, s=45, color=NAVY, zorder=4)
    ax.text(origin[0] - 0.05, origin[1] - 0.42, "基线谱 b", ha="center", color=NAVY, weight="bold")
    ax.text(3.72, 3.18, "未保护候选 Xc", color=VERMILION, weight="bold")
    ax.text(projected[0] + 0.08, projected[1] - 0.18, "锥投影 Xcone", color=GREEN, weight="bold")
    ax.text(3.55, 0.12, "保留平行 / common-shading 分量", color=NAVY, ha="center")
    ax.text(5.36, 1.90, "缩放谱正交分量", color=ORANGE, ha="left", rotation=90)
    ax.add_patch(
        Arc(
            origin,
            2.0,
            1.25,
            angle=0,
            theta1=0,
            theta2=18,
            color=GREEN,
            linewidth=1.4,
        )
    )
    ax.text(1.33, 0.36, f"固定半角 {angle:.1f}°", color=GREEN, weight="bold")
    ax.text(
        0.03,
        0.95,
        "光谱锥的几何作用（角度示意不按比例）",
        transform=ax.transAxes,
        fontsize=13,
        weight="bold",
        color=NAVY,
        va="top",
    )

    ax = axes[1]
    scenes = ["3DSSZ", "ZKH3"]
    colors = [BLUE, ORANGE]
    bars = ax.bar(scenes, clip, color=colors, width=0.56)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("cone clip fraction")
    ax.set_title("裁剪比例必须作为风险量公开", color=NAVY, fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, clip, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color=NAVY,
        )
    fig.suptitle(
        f"固定 {angle:.1f}° spectral cone：限制谱角偏移，但不把被保留的高频自动变成真值",
        fontsize=16.5,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "高裁剪率说明原始 ridge 候选经常离开窄谱锥；锥投影是保护项，不是 RGB 高频真实性证明。",
        ha="center",
        fontsize=9.5,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.94), w_pad=2.4)
    return save_figure(
        fig,
        output_dir,
        "05_spectral_cone_geometry",
        alt=(
            f"固定 {angle:.1f} 度光谱锥的几何示意与两场景裁剪比例柱图；"
            f"3DSSZ 裁剪 {clip[0]:.4f}，ZKH3 裁剪 {clip[1]:.4f}。"
        ),
        claim_scope=(
            "光谱锥只限制候选相对 bicubic 基线的谱正交分量并公开裁剪风险；"
            "它不证明保留的 common-shading 高频来自真实 SWIR。"
        ),
        sources=["v8_wald_benchmark.json", "v8_wald_summary.csv"],
    )


def _sweep_family(label: str) -> str:
    lowered = label.lower()
    if lowered.startswith("v7") or lowered.startswith("coeff"):
        return "V7 / coefficient"
    if "gain" in lowered or "additive" in lowered:
        return "gain / additive"
    if "simplex" in lowered or "hybrid" in lowered:
        return "simplex / hybrid"
    if "bridge" in lowered:
        return "low-rank bridge"
    return "other"


def figure_06_pareto(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    data = evidence.detail_sweep.copy().reset_index(drop=True)
    data["family"] = data["label"].astype(str).map(_sweep_family)
    pass_count = int(data["all_internal_bounds"].fillna(False).astype(bool).sum())
    total = len(data)
    family_style = {
        "V7 / coefficient": (BLUE, "o"),
        "gain / additive": (ORANGE, "s"),
        "simplex / hybrid": (PURPLE, "D"),
        "low-rank bridge": (GREEN, "^"),
        "other": (GRAY, "x"),
    }
    fig = plt.figure(figsize=(9.0, 6.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.15, 1.15), wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    v7 = data[data["label"] == "V7 frozen"].iloc[0]
    ax.axvline(float(v7["R_perp_2201"]), color=GRID, lw=1.0, ls="--")
    ax.axhline(float(v7["beta_2201"]), color=GRID, lw=1.0, ls="--")
    for family, (color, marker) in family_style.items():
        frame = data[data["family"] == family]
        if frame.empty:
            continue
        ax.scatter(
            frame["R_perp_2201"],
            frame["beta_2201"],
            s=72,
            marker=marker,
            color=color,
            edgecolor=WHITE if marker != "x" else color,
            linewidth=0.8,
            label=family,
            zorder=3,
        )
        for index, row in frame.iterrows():
            ax.annotate(
                str(index + 1),
                (float(row["R_perp_2201"]), float(row["beta_2201"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=DARK_GRAY,
            )
    coefficient_labels = ["V7 frozen", "coeff 0.020", "coeff 0.050", "coeff 0.100"]
    coefficient = data.set_index("label").loc[coefficient_labels]
    ax.plot(
        coefficient["R_perp_2201"],
        coefficient["beta_2201"],
        color=BLUE,
        lw=1.2,
        alpha=0.75,
        zorder=2,
    )
    ax.annotate(
        "期望方向：β 提高且 R⊥ 降低",
        xy=(0.06, 0.92),
        xytext=(0.27, 0.78),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4),
        color=GREEN,
        weight="bold",
        fontsize=10,
    )
    ax.set_xlabel("2201 nm 非相干高频 R⊥（低为优）")
    ax.set_ylabel("2201 nm 相干细节幅值 β（仅显示诊断）")
    ax.set_title("19 组真实 3DSSZ：幅值提升伴随伪影或其他约束恶化", color=NAVY, fontsize=13)
    ax.grid(color=GRID, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8.7)

    side = fig.add_subplot(grid[0, 1])
    side.axis("off")
    add_box(
        side,
        (0.04, 0.80),
        0.92,
        0.15,
        f"联合筛查：{pass_count}/{total} 通过",
        "没有一个候选同时满足\n现有全部内部边界",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        title_size=14,
        body_size=9.2,
    )
    plateau = data[data["label"] == "plateau gate + gain 1.40"].iloc[0]
    coefficient_high = data[data["label"] == "coeff 0.100"].iloc[0]
    add_box(
        side,
        (0.04, 0.61),
        0.92,
        0.14,
        "公共 gain 能推高 β，但不是 Pareto 解",
        f"plateau + gain 1.40：β={float(plateau['beta_2201']):.3f}\n"
        f"R⊥={float(plateau['R_perp_2201']):.3f}；暗区 β={float(plateau['dark_beta_2201']):.3f}",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        body_size=8.6,
    )
    add_box(
        side,
        (0.04, 0.44),
        0.92,
        0.12,
        "放大 coefficient 的收益很小",
        f"coeff 0.100：β={float(coefficient_high['beta_2201']):.3f}，"
        f"R⊥={float(coefficient_high['R_perp_2201']):.3f}",
        face=PALE_BLUE,
        edge=BLUE,
        body_size=8.8,
    )
    bridge_best = data[data["family"] == "low-rank bridge"].sort_values("forward_rmse").iloc[0]
    add_box(
        side,
        (0.04, 0.22),
        0.92,
        0.15,
        "simplex / bridge 也未形成终解",
        f"最佳 bridge：forward RMSE={float(bridge_best['forward_rmse']):.5f}\n"
        f"2201 nm β={float(bridge_best['beta_2201']):.3f}；simplex abundance\n"
        "在 2201 nm 出现对比反转。",
        face=LIGHT,
        edge=GRAY,
        title_color=NAVY,
        body_size=7.2,
    )
    side.text(
        0.04,
        0.13,
        "散点编号仅用于区分全部 19 个运行；\n颜色 / 形状表示方案家族，关键数值取自冻结 CSV。",
        transform=side.transAxes,
        fontsize=8.3,
        color=DARK_GRAY,
        linespacing=1.4,
    )

    fig.suptitle(
        "真实场景失败证据：继续扫 strength 无法把 RGB 相干幅值与非相干高频解耦",
        fontsize=17,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.014,
        "β≈1 已降为显示诊断而非科学硬目标；即便如此，19 组结果仍不支持继续放大公共 gain、PCA residual 或 additive。",
        ha="center",
        fontsize=9.4,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(
        fig,
        output_dir,
        "06_3dssz_pareto_failure",
        alt=(
            f"19 组真实 3DSSZ 方案的 2201 nm 相干幅值 beta 与非相干高频 R-perp 散点图；"
            f"通过全部内部筛查的方案为 {pass_count}/{total}。"
        ),
        claim_scope=(
            "同数据 RGB 引导的内部失败诊断；证明继续扫公共强度未形成联合 Pareto 点，"
            "但 beta 本身不是独立 HR-SWIR 真值。"
        ),
        sources=["v8_detail_sweep.csv", "v8_detail_sweep_interpretation.md"],
    )


def figure_07_registration(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.7), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.7, 2.7)
    ax.axhline(0.0, color=GRID, lw=0.9)
    ax.axvline(0.0, color=GRID, lw=0.9)
    ellipse = Ellipse(
        (0.0, 0.0),
        width=5.2,
        height=2.5,
        angle=28.0,
        facecolor=PALE_BLUE,
        edgecolor=BLUE,
        linewidth=1.7,
        alpha=0.9,
    )
    ax.add_patch(ellipse)
    theta = math.radians(28.0)
    major = np.asarray([math.cos(theta), math.sin(theta)]) * 2.15
    minor = np.asarray([-math.sin(theta), math.cos(theta)]) * 0.92
    points = np.stack(
        [
            np.asarray([0.0, 0.0]),
            major,
            -major,
            minor,
            -minor,
        ]
    )
    labels = [r"$\mu_0$", r"$\mu_{+1}$", r"$\mu_{-1}$", r"$\mu_{+2}$", r"$\mu_{-2}$"]
    for index, (point, label) in enumerate(zip(points, labels, strict=True)):
        color = NAVY if index == 0 else ORANGE
        marker = "o" if index == 0 else "D"
        ax.scatter(point[0], point[1], s=85, color=color, marker=marker, zorder=4)
        ax.text(
            point[0] + 0.13,
            point[1] + 0.13,
            label,
            color=color,
            weight="bold",
            fontsize=10,
        )
    ax.annotate(
        "主轴方向",
        xy=major * 0.82,
        xytext=(0.2, 1.75),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2),
        color=BLUE,
        fontsize=9.5,
    )
    ax.set_xlabel("局部 warp 参数主方向 1（示意）")
    ax.set_ylabel("局部 warp 参数主方向 2（示意）")
    ax.set_title("局部注册协方差代理 Σφ 与五点传播", color=NAVY, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.02,
        "椭圆与点位仅示意；尚无独立地标标定的真实 Σφ。",
        transform=ax.transAxes,
        fontsize=8.7,
        color=VERMILION,
        weight="bold",
    )

    ax = axes[1]
    ax.axis("off")
    ax.text(
        0.02,
        0.94,
        "每个 sigma point 都必须穿过完整传感器链",
        transform=ax.transAxes,
        fontsize=13.2,
        color=NAVY,
        weight="bold",
        va="top",
    )
    add_box(
        ax,
        (0.03, 0.69),
        0.25,
        0.15,
        "局部配准估计",
        "score / margin\nbackward error",
        face=PALE_BLUE,
        edge=BLUE,
        title_size=9.2,
        body_size=7.2,
    )
    add_box(
        ax,
        (0.37, 0.69),
        0.25,
        0.15,
        "五个 warp 假设",
        "中心点 + 两主轴\n正负扰动",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        title_size=9.2,
        body_size=7.2,
    )
    add_box(
        ax,
        (0.71, 0.69),
        0.25,
        0.15,
        "原生 forward",
        "warp → PSF/MTF\n采样 / mask / noise",
        face=PALE_GREEN,
        edge=GREEN,
        title_size=9.2,
        body_size=7.0,
    )
    add_arrow(ax, (0.28, 0.765), (0.37, 0.765), color=BLUE)
    add_arrow(ax, (0.62, 0.765), (0.71, 0.765), color=ORANGE)

    add_box(
        ax,
        (0.24, 0.50),
        0.52,
        0.12,
        "传播统计",
        "汇总五次 forward 的残差方差与细节符号稳定性",
        face=LIGHT,
        edge=GRAY,
        title_size=9.2,
        body_size=7.2,
    )
    add_arrow(ax, (0.835, 0.69), (0.64, 0.62), color=GRAY, connectionstyle="arc3,rad=0.15")

    add_box(
        ax,
        (0.04, 0.27),
        0.41,
        0.15,
        "稳定 → c_reg(p)",
        "跨五点相位 / 符号稳定\n且 forward 残差非劣",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        title_size=9.2,
        body_size=7.2,
    )
    add_box(
        ax,
        (0.55, 0.27),
        0.41,
        0.15,
        "不稳定 → fail-closed",
        "关闭 RGB 注入\n回退 observation-only",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        title_size=9.2,
        body_size=7.2,
    )
    add_arrow(ax, (0.45, 0.50), (0.28, 0.42), color=GREEN, connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.55, 0.50), (0.72, 0.42), color=VERMILION, connectionstyle="arc3,rad=-0.12")
    add_box(
        ax,
        (0.04, 0.07),
        0.92,
        0.12,
        "发布前必须补齐",
        "独立地标 + 重复采集 + 局部残差模型 → 冻结 Σφ；\n当前 tie-point 分数只能构造 proxy。",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        title_size=9.0,
        body_size=7.0,
    )
    fig.suptitle(
        "注册不确定性进入融合：从单个 warp 点估计升级为局部协方差传播",
        fontsize=16.8,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "这是待实现与标定的 uncertainty proxy 合同，不是当前真实配准精度或协方差测量结果。",
        ha="center",
        fontsize=9.4,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.94), w_pad=2.0)
    return save_figure(
        fig,
        output_dir,
        "07_registration_uncertainty_sigma_points",
        alt=(
            "局部注册协方差椭圆和五个 sigma point 的示意，以及每个点经完整原生 forward 链传播后形成稳定性门控或 fail-closed 回退的流程。"
        ),
        claim_scope=(
            "待实现的注册不确定性代理与传播合同；椭圆非实测，当前不得据此声称真实亚像素精度或已标定协方差。"
        ),
        sources=["uarf_cycle_formula_audit.md", "v8_method_decision.md"],
    )


def _contract_column(
    ax: plt.Axes,
    x: float,
    width: float,
    title: str,
    subtitle: str,
    rows: list[tuple[str, str]],
    *,
    face: str,
    edge: str,
    warning: str,
) -> None:
    add_box(
        ax,
        (x, 0.67),
        width,
        0.14,
        title,
        subtitle,
        face=face,
        edge=edge,
        title_color=edge,
        title_size=10.7,
        body_size=7.2,
    )
    row_height = 0.085
    y = 0.55
    for label, body in rows:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                row_height,
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=GRID,
                linewidth=0.9,
            )
        )
        ax.text(
            x + 0.014,
            y + row_height / 2,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.5,
            weight="bold",
            color=NAVY,
        )
        ax.text(
            x + width * 0.30,
            y + row_height / 2,
            textwrap.fill(body, width=29, break_long_words=True),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.8,
            color=DARK_GRAY,
            linespacing=1.15,
        )
        y -= row_height
    add_box(
        ax,
        (x, 0.055),
        width,
        0.10,
        "边界",
        warning,
        face=PALE_RED if edge == ORANGE else LIGHT,
        edge=VERMILION if edge == ORANGE else GRAY,
        title_color=VERMILION,
        title_size=8.0,
        body_size=6.7,
    )


def figure_08_product_contract(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    angle = float(evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"])
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "双产品合同：从文件级彻底分开“科学可用”与“肉眼更像 RGB”",
        transform=ax.transAxes,
        fontsize=14.5,
        weight="bold",
        color=NAVY,
        va="top",
    )
    add_box(
        ax,
        (0.29, 0.835),
        0.42,
        0.075,
        "共同起点：原生观测 + gate map + observation-only X0",
        "同一配准与元数据；从这里开始分叉并分别落盘",
        face=LIGHT,
        edge=GRAY,
        title_size=8.7,
        body_size=6.7,
    )
    add_arrow(ax, (0.47, 0.835), (0.25, 0.81), color=GREEN, connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.53, 0.835), (0.75, 0.81), color=ORANGE, connectionstyle="arc3,rad=-0.12")

    scientific_rows = [
        ("纹理来源", "gate ON 的受约束低秩细节；其余区域使用 X0"),
        ("光谱保护", f"固定 {angle:.1f}° cone + native forward"),
        ("必带证据", "gate / cone clip / forward residual / uncertainty"),
        ("允许用途", "谱形、吸收深度、矿物识别与定量指标"),
        ("文件合同", "独立 scientific 目录；不可被可视化覆盖"),
    ]
    visual_rows = [
        ("纹理来源", "显式 RGB 高频纹理化；不伪装成传感器观测"),
        ("光谱保护", "以 scientific cube 为底图；外观层不进光谱分析"),
        ("必带证据", "manifest 标志、独立文件名与显式警示"),
        ("允许用途", "岩心肉眼判读、交流展示与制图"),
        ("文件合同", "独立 visualization 目录；永不参与 RMSE / SAM"),
    ]
    _contract_column(
        ax,
        0.03,
        0.44,
        "Scientific fusion cube",
        "保守、可审计、允许明确不恢复",
        scientific_rows,
        face=PALE_GREEN,
        edge=GREEN,
        warning="没有独立 HR 真值时，不宣称真实 SWIR 细节无损。",
    )
    _contract_column(
        ax,
        0.53,
        0.44,
        "RGB-textured visualization",
        "可追求接近 RGB 的观感，但必须显式标注",
        visual_rows,
        face=PALE_ORANGE,
        edge=ORANGE,
        warning="RGB-textured visualization\nnot for quantitative spectroscopy",
    )
    fig.text(
        0.5,
        0.015,
        "视觉结果“接近 RGB”可以是展示目标；科学立方体只恢复可辨识的共有结构，两者不得混用、覆盖或共用结论。",
        ha="center",
        fontsize=9.7,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(
        fig,
        output_dir,
        "08_scientific_vs_visualization_contract",
        alt=(
            "Scientific fusion cube 与 RGB-textured visualization 的双产品对照，逐项比较纹理来源、光谱保护、证据、允许用途和文件合同。"
        ),
        claim_scope=(
            "产品治理和论文声明边界；允许可视化接近 RGB，但禁止其进入定量光谱、矿物识别或真实 SWIR 细节恢复结论。"
        ),
        sources=["v8_method_decision.md", "v8_band_specific_detail_design.md"],
    )


def _extract_warp_sweep(text: str) -> str:
    match = re.search(r"warp sweep（至少\s*([^）]+)）", text)
    return match.group(1).strip() if match else "预注册多级亚像素扰动"


def figure_09_validation_matrix(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    warp_levels = _extract_warp_sweep(evidence.method_decision_text)
    rows = [
        (
            "候选无关\n可辨识性",
            "同 MTF log-DoG、空间留块、RGB shift-null",
            "低频相关被误当作高频许可",
            "已完成｜两场景",
            "冻结阈值后到独立岩心复测",
            "complete",
        ),
        (
            "严格 Wald 伪真值",
            "候选冻结后才读取 pseudo-HR HSI truth",
            "参数泄漏或只看观感",
            "已完成｜内部",
            "跨场景 RMSE/SAM 与目标波段均不劣",
            "complete",
        ),
        (
            "真实 3DSSZ Pareto",
            "19 组细节通道；β、R⊥、flat/dark、forward",
            "更大 gain 被误判为真实细节",
            "已完成｜0/19 通过",
            "停止继续盲扫公共 strength",
            "complete",
        ),
        (
            "注册扰动",
            f"warp sweep：{warp_levels}；局部 covariance 五点传播",
            "轻微错位仍复制 RGB 边缘",
            "待独立标定",
            "相位/符号和收益在预注册容差内稳定",
            "pending",
        ),
        (
            "dark / flat 噪声注入",
            "dark-current、shot/read noise、平场与低对比阶跃",
            "暗区噪声被增强成纹理",
            "缺 dark/flat",
            "暗区改善且 flat HF / R⊥ 不恶化",
            "missing",
        ),
        (
            "RGB shuffle / 大幅环移",
            "打乱 RGB、场景内环移、保持 HSI 不变",
            "模型无条件复制 RGB",
            "gate shift-null 已有；产品级待做",
            "反事实输出回退 observation-only",
            "partial",
        ),
        (
            "模态专属边缘",
            "RGB-only 标记 / 阴影 / 高光与 HSI-only 吸收边缘",
            "把模态专属边缘写入全部波段",
            "缺独立标注",
            "RGB-only edge 不注入；HSI-only 不被抹除",
            "missing",
        ),
        (
            "独立 HR-\nNIR/SWIR truth",
            "高分辨局部拍摄\n微位移超采样\n受控分辨率靶",
            "内部一致性冒充\n真实高频恢复",
            "尚缺",
            "冻结方法在留出岩心上通过真值指标",
            "missing",
        ),
    ]
    columns = ["验证 / 反事实", "控制变量", "要证伪的问题", "当前状态", "进入 / 停止条件"]
    widths = np.asarray([0.16, 0.25, 0.22, 0.15, 0.22], dtype=float)
    widths /= widths.sum()
    fig, ax = plt.subplots(figsize=(8.0, 10.2))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "论文验证与反事实矩阵：每项实验都必须能推翻一种“看起来更清晰”的错误解释",
        transform=ax.transAxes,
        fontsize=15.5,
        color=NAVY,
        weight="bold",
        va="top",
    )
    left, right, bottom, top = 0.02, 0.985, 0.075, 0.89
    table_width = right - left
    header_height = 0.065
    row_height = (top - bottom - header_height) / len(rows)
    x_positions = left + table_width * np.concatenate(([0.0], np.cumsum(widths[:-1])))
    cell_widths = table_width * widths
    for label, x, width in zip(columns, x_positions, cell_widths, strict=True):
        ax.add_patch(
            Rectangle(
                (x, top - header_height),
                width,
                header_height,
                transform=ax.transAxes,
                facecolor=NAVY,
                edgecolor=WHITE,
                linewidth=1.0,
            )
        )
        ax.text(
            x + width / 2,
            top - header_height / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=WHITE,
            fontsize=8.2,
            weight="bold",
        )
    status_faces = {
        "complete": PALE_GREEN,
        "partial": PALE_ORANGE,
        "pending": PALE_BLUE,
        "missing": PALE_RED,
    }
    status_edges = {
        "complete": GREEN,
        "partial": ORANGE,
        "pending": BLUE,
        "missing": VERMILION,
    }
    wrap_widths = [9, 17, 14, 10, 16]
    for row_index, row in enumerate(rows):
        values = row[:-1]
        status = row[-1]
        y = top - header_height - (row_index + 1) * row_height
        for column_index, (value, x, width) in enumerate(
            zip(values, x_positions, cell_widths, strict=True)
        ):
            face = status_faces[status] if column_index in (0, 3) else (WHITE if row_index % 2 == 0 else LIGHT)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    row_height,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=WHITE,
                    linewidth=1.0,
                )
            )
            ax.text(
                x + width / 2,
                y + row_height / 2,
                textwrap.fill(
                    str(value),
                    width=wrap_widths[column_index],
                    break_long_words=True,
                    break_on_hyphens=False,
                ),
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=status_edges[status] if column_index in (0, 3) else DARK_GRAY,
                weight="bold" if column_index in (0, 3) else "normal",
                fontsize=6.8,
                linespacing=1.12,
            )
    policy = evidence.wald_report["fail_closed_policy"]["decisions"]
    ax.text(
        0.02,
        0.025,
        f"当前解释边界：3DSSZ gate {str(policy['3dssz']['gate_state']).upper()}；ZKH3 仅 conditional；两场景不构成分类器验证。\n"
        "缺任一关键证据时，scientific 输出回退 observation-only。",
        transform=ax.transAxes,
        fontsize=8.0,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(
        fig,
        output_dir,
        "09_validation_counterfactual_matrix",
        alt=(
            "论文验证与反事实矩阵，列出 identifiability、Wald、真实 Pareto、注册扰动、暗平场噪声、RGB shuffle、模态专属边缘和独立 HR 真值的控制变量、证伪目标、状态及验收条件。"
        ),
        claim_scope=(
            "预注册实验合同与当前证据缺口；完成项仅限现有内部证据，待办项不得提前包装为已验证结果。"
        ),
        sources=["v8_method_decision.md", "v8_wald_benchmark.json", "v8_detail_sweep.csv"],
    )


def figure_10_truth_roadmap(
    evidence: Evidence, output_dir: Path
) -> dict[str, Any]:
    angle = float(evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"])
    stages = [
        (
            "A  标定",
            "band-wise MTF/PSF\ndark-flat / noise\n独立地标与 Σφ",
            PALE_BLUE,
            BLUE,
        ),
        (
            "B  真值",
            "HR-NIR / HR-SWIR\n微位移超采样\n分辨率 / 暗阶跃靶",
            PALE_GREEN,
            GREEN,
        ),
        (
            "C  冻结",
            "独立岩心分层\ntrain / val / test 隔离\n退化协议冻结",
            LIGHT,
            GRAY,
        ),
        (
            "D  终验",
            f"band-pass gate\nrank-8 + {angle:.1f}° cone\nforward + 反事实",
            PALE_ORANGE,
            ORANGE,
        ),
        (
            "E  投稿",
            "双产品分离\n失败 ROI / CI 公开\n声明由真值决定",
            PALE_GREEN,
            NAVY,
        ),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "独立真值采集与 V8 实施路线：先补信息，再决定模型复杂度",
        transform=ax.transAxes,
        fontsize=15.5,
        color=NAVY,
        weight="bold",
        va="top",
    )
    add_box(
        ax,
        (0.02, 0.72),
        0.14,
        0.15,
        "当前位置",
        "两场景内部证据\n3DSSZ OFF\nZKH3 conditional",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=7.0,
    )
    x_positions = np.linspace(0.20, 0.82, len(stages))
    width = 0.145
    for index, (x, stage) in enumerate(zip(x_positions, stages, strict=True)):
        title, body, face, edge = stage
        add_box(
            ax,
            (float(x), 0.67),
            width,
            0.22,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
            title_size=8.2,
            body_size=6.7,
        )
        if index == 0:
            add_arrow(ax, (0.16, 0.795), (float(x), 0.795), color=VERMILION)
        else:
            add_arrow(
                ax,
                (float(x_positions[index - 1]) + width, 0.795),
                (float(x), 0.795),
                color=BLUE,
            )

    lane_y = [0.47, 0.31, 0.15]
    lane_titles = ["数据 / 标定交付", "算法判定", "停止与进入条件"]
    lane_texts = [
        [
            "PSF/MTF、noise、Σφ",
            "独立 HR truth / 靶标",
            "冻结 split / protocol",
            "QA / 反事实记录",
            "数据 / 代码索引",
        ],
        [
            "构造原生 Dmq",
            "仅评价，不回流调参",
            "冻结 gate 与候选",
            "逐 band / region 关闭",
            "主张 / 证据映射",
        ],
        [
            "标定不全 → 不开闸",
            "无真值 → 禁止“无损”",
            "数据泄漏 → 重做协议",
            "负收益 / 伪影 → 回退 X0",
            "仅通过后进入投稿",
        ],
    ]
    for lane_index, (y, lane_title, texts) in enumerate(
        zip(lane_y, lane_titles, lane_texts, strict=True)
    ):
        ax.text(
            0.02,
            y + 0.045,
            lane_title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.4,
            weight="bold",
            color=NAVY if lane_index < 2 else VERMILION,
        )
        ax.plot(
            [0.18, 0.965],
            [y, y],
            transform=ax.transAxes,
            color=GRID,
            lw=1.0,
            zorder=0,
        )
        for x, text_value, stage in zip(x_positions, texts, stages, strict=True):
            _, _, face, edge = stage
            ax.add_patch(
                FancyBboxPatch(
                    (float(x), y - 0.035),
                    width,
                    0.075,
                    boxstyle="round,pad=0.008,rounding_size=0.012",
                    transform=ax.transAxes,
                    facecolor=PALE_RED if lane_index == 2 else face,
                    edgecolor=VERMILION if lane_index == 2 else edge,
                    linewidth=1.0,
                )
            )
            ax.text(
                float(x) + width / 2,
                y + 0.002,
                text_value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.6,
                color=VERMILION if lane_index == 2 else DARK_GRAY,
                weight="bold" if lane_index == 2 else "normal",
            )
    ax.text(
        0.02,
        0.055,
        "复杂度升级条件：冻结低秩分支须先在独立真值上稳定受益，且表达能力成为明确瓶颈；\n"
        "此后才评估 active-simplex 或小型残差网络。",
        transform=ax.transAxes,
        fontsize=7.7,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(
        fig,
        output_dir,
        "10_truth_acquisition_roadmap",
        alt=(
            "从仪器与注册标定、独立 HR 真值采集、数据协议冻结、条件算法终验到论文双产品交付的五阶段实施路线，并列出每阶段的交付、算法判定和停止条件。"
        ),
        claim_scope=(
            "未来实施与证据获取路线；当前仅处于两场景内部证据阶段，后续阶段和论文声明均以独立真值通过为前提。"
        ),
        sources=["v8_method_decision.md", "uarf_cycle_formula_audit.md"],
    )


def _validate_figure_files(output_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(FIGURE_STEMS)
    actual = {str(entry["stem"]) for entry in entries}
    if actual != expected:
        raise RuntimeError(f"Figure stem mismatch: expected={sorted(expected)}, actual={sorted(actual)}")
    checks: list[dict[str, Any]] = []
    for stem in FIGURE_STEMS:
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        if not png.exists() or not pdf.exists():
            raise FileNotFoundError(f"Missing figure pair for {stem}")
        with Image.open(png) as image:
            image.verify()
        with Image.open(png) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0.0, 0.0))
        if width < 1200 or height < 800:
            raise RuntimeError(f"Figure {stem} is too small: {width}x{height}")
        if not (295.0 <= float(dpi[0]) <= 305.0 and 295.0 <= float(dpi[1]) <= 305.0):
            raise RuntimeError(f"Figure {stem} does not report 300 dpi: {dpi}")
        if pdf.read_bytes()[:4] != b"%PDF":
            raise RuntimeError(f"Invalid PDF header: {pdf}")
        checks.append(
            {
                "stem": stem,
                "png_readable": True,
                "pixel_size": [int(width), int(height)],
                "dpi": [round(float(dpi[0]), 2), round(float(dpi[1]), 2)],
                "pdf_header_valid": True,
            }
        )
    return {"passed": True, "figure_count": len(checks), "checks": checks}


def _manifest(
    repo: Path,
    evidence: Evidence,
    entries: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    policy = evidence.wald_report["fail_closed_policy"]
    ordered = sorted(entries, key=lambda item: FIGURE_STEMS.index(str(item["stem"])))
    source_files = {
        name: str(path.resolve().relative_to(repo.resolve())).replace("\\", "/")
        for name, path in evidence.source_paths.items()
    }
    return {
        "schema": "GeoCoreFusion-V8-publication-figures-v1",
        "output_contract": {
            "png_dpi": 300,
            "paired_vector_pdf": True,
            "style": "V5/V6-derived navy-blue, sparse warning orange, Chinese sans-serif",
        },
        "policy_snapshot": {
            "status": policy["policy_status"],
            "3dssz_gate": policy["decisions"]["3dssz"]["gate_state"],
            "zkh3_gate": policy["decisions"]["zkh3"]["gate_state"],
            "scene_count": len(policy["decisions"]),
            "spectral_cone_half_angle_deg": float(
                evidence.wald_report["protocol"]["spectral_cone_half_angle_deg"]
            ),
            "independent_data_revalidation_required": bool(
                policy["independent_data_revalidation_required"]
            ),
        },
        "figures": ordered,
        "source_files": source_files,
        "truth_scope": (
            "Wald truth is internal pseudo-HR HSI; identifiability and real-scene detail metrics are same-data diagnostics. "
            "No figure claims independent HR-SWIR truth, released V8 software, or lossless SWIR detail recovery."
        ),
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="GeoCoreFusion repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/v8_research/figures)",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo / "artifacts" / "v8_research" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    evidence = read_evidence(repo)

    entries: list[dict[str, Any]] = []
    lightweight_builders: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("01", lambda: figure_01_route(evidence, output_dir)),
        ("02", lambda: figure_02_identifiability(evidence, output_dir)),
        ("03", lambda: figure_03_wald_methods(evidence, output_dir)),
        ("05", lambda: figure_05_cone(evidence, output_dir)),
        ("06", lambda: figure_06_pareto(evidence, output_dir)),
        ("07", lambda: figure_07_registration(evidence, output_dir)),
        ("08", lambda: figure_08_product_contract(evidence, output_dir)),
        ("09", lambda: figure_09_validation_matrix(evidence, output_dir)),
        ("10", lambda: figure_10_truth_roadmap(evidence, output_dir)),
    ]
    for number, builder in lightweight_builders:
        print(f"building figure {number}", flush=True)
        entries.append(builder())

    # Build the memory-heavier Wald visual last so the other nine figures are
    # already available if the host needs resource troubleshooting.
    print("building figure 04", flush=True)
    entries.append(figure_04_wald_visual(repo, evidence, output_dir))

    validation = _validate_figure_files(output_dir, entries)
    manifest = _manifest(repo, evidence, entries, validation)
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(reloaded.get("figures", [])) != len(FIGURE_STEMS):
        raise RuntimeError("Manifest figure count validation failed")
    print(
        json.dumps(
            {
                "figures": len(FIGURE_STEMS),
                "png_pdf_files": len(FIGURE_STEMS) * 2,
                "manifest": str(manifest_path),
                "validation_passed": validation["passed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
