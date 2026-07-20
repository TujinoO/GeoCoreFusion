"""Build publication-ready V6.1 method, comparison, and evidence figures.

The real-scene comparison reconstructs 901/1651/2201 nm reflectance from the
saved material coefficients and applies one shared display mapping to V6 and
V6.1 within each scene.  The pipeline's independently stretched preview PNGs
are used only as a reconstruction checksum, never as the comparison domain.

No source run is modified.  PNG and vector PDF versions are written for every
figure, together with a machine-readable benchmark summary and manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_v7_figures as base  # noqa: E402


NAVY = "#16324F"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
DARK_GRAY = "#374151"
GRID = "#D8DEE8"
LIGHT = "#F5F7FA"
PALE_BLUE = "#E5F1F8"
PALE_GREEN = "#E5F5EF"
PALE_ORANGE = "#FFF1D6"
PALE_RED = "#FCE8E2"
WHITE = "#FFFFFF"

METHODS = ("V6", "V6.1")
METHOD_COLORS = {"V6": GRAY, "V6.1": BLUE}
SCENES: dict[str, dict[str, Any]] = {
    "3dssz": {
        "label": "3DSSZ",
        "runs": {
            "V6": "3dssz_roi_fusion_v6_v7eval_ampfix",
            "V6.1": "3dssz_roi_fusion_v61_benchmark",
        },
        "detail_crop": (290, 960, 260, 320),
        "dark_crop": (520, 180, 260, 320),
    },
    "zkh3": {
        "label": "ZKH3",
        "runs": {
            "V6": "zkh3_roi_fusion_v6_v7eval_ampfix",
            "V6.1": "zkh3_roi_fusion_v61_benchmark",
        },
        "detail_crop": (555, 545, 260, 320),
        "dark_crop": (95, 455, 260, 320),
    },
}


def setup_style() -> None:
    base.setup_style()
    plt.rcParams.update(
        {
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png_path.name, pdf_path.name]


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
    title_size: float = 10.2,
    body_size: float = 8.5,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.71,
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
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        color=DARK_GRAY,
        fontsize=body_size,
        linespacing=1.32,
        transform=ax.transAxes,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    width: float = 1.6,
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
        )
    )


def fusion_config_from_payload(payload: dict[str, Any], fusion_config_type: Any) -> Any:
    config = fusion_config_type()
    allowed = {field.name for field in fields(config)}
    for key, value in payload.items():
        if key not in allowed:
            continue
        current = getattr(config, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    return config


def selected_band_payload(report: dict[str, Any], wavelength: str) -> dict[str, Any]:
    return report["spatial"]["band_detail_by_brightness"]["bands"][wavelength]


def extract_metrics(run_dir: Path) -> dict[str, Any]:
    report = read_json(run_dir / "metrics" / "quality_report.json")
    values: dict[str, Any] = {
        "status": report["summary"]["status"],
        "boundary_correlation": float(
            report["spatial"]["rgb_material_boundary_correlation"]
        ),
        "gain_lowres_rmse": float(
            report["spatial"]["detail_gain_lowres_rmse_from_one"]
        ),
        "additive_lowres_rmse": float(
            report["spatial"]["additive_detail_lowres_rmse_from_zero"]
        ),
        "forward_rmse": float(report["final_hr_product_observation"]["rmse"]),
        "forward_sam_deg": float(
            report["final_hr_product_observation"]["sam_mean_deg"]
        ),
        "forward_band_cc": float(
            report["final_hr_product_observation"]["band_cc_mean"]
        ),
        "registration": {},
        "bands": {},
    }
    roi = report["registration"]["roi_refinement"]
    for sensor in ("nir", "swir"):
        sensor_payload = roi[sensor]
        tiepoints = sensor_payload.get("details", {}).get("tiepoint_refinement", {})
        values["registration"][sensor] = {
            "score_before": float(sensor_payload["score_before"]),
            "score_after": float(sensor_payload["score_after"]),
            "tie_point_count": int(tiepoints.get("tie_point_count", 0)),
            "accepted_affine": bool(sensor_payload.get("accepted_affine", False)),
        }
    for wavelength in ("901.0nm", "1651.0nm", "2201.0nm"):
        band = selected_band_payload(report, wavelength)
        scale = band["multiscale_log_high_frequency"]["sigma_2.4px"]
        edge = band["gradient_and_edge"]["reliable_rgb_detail"]
        halo = band["halo_and_edge_spread_proxy"]
        values["bands"][wavelength] = {
            "all_valid": scale["all_valid"],
            "darkest_percentile": scale["darkest_percentile"],
            "reliable_rgb_detail": scale["reliable_rgb_detail"],
            "dark_reliable_rgb_detail": scale["dark_reliable_rgb_detail"],
            "rgb_flat": scale["rgb_flat"],
            "gradient_orientation_coherence": float(
                edge["gradient_orientation_coherence_abs_cosine"]
            ),
            "edge_f1_1px": float(edge["edge_f1_1px"]),
            "edge_width_ratio": float(halo["edge_width_ratio"]),
            "halo_status": str(halo["status"]),
        }
    return values


def collect_data(repo: Path, api: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scene, spec in SCENES.items():
        scene_result: dict[str, Any] = {"methods": {}}
        v61_dir = repo / "runs" / spec["runs"]["V6.1"]
        rgb = base.load_rgb_crop(repo, v61_dir, api)
        for method in METHODS:
            run_dir = repo / "runs" / spec["runs"][method]
            required = [
                run_dir / "analysis" / "harmonized_lowres.hdr",
                run_dir / "coefficients" / "material_coefficients.hdr",
                run_dir / "metrics" / "spatial_detail_gain.hdr",
                run_dir / "metrics" / "spatial_additive_detail.hdr",
                run_dir / "metrics" / "quality_report.json",
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete figure source run {run_dir.name}; missing {missing}"
                )
            image, _, diagnostics = base.reconstruct_display_bands(
                repo, run_dir, api, rgb_cache=rgb
            )
            scene_result["methods"][method] = {
                "run": run_dir.name,
                "run_dir": run_dir,
                "image": image,
                "metrics": extract_metrics(run_dir),
                "diagnostics": diagnostics,
            }
        images = [scene_result["methods"][method]["image"] for method in METHODS]
        scene_result["rgb"] = rgb
        scene_result["display_bounds"] = base.percentile_bounds(images)
        scene_result["detail_crop"] = tuple(int(v) for v in spec["detail_crop"])
        scene_result["dark_crop"] = tuple(int(v) for v in spec["dark_crop"])
        result[scene] = scene_result
    return result


def method_route(output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(18.5, 9.3))
    ax.axis("off")
    ax.text(
        0.02,
        0.96,
        "GeoCoreFusion V6.1：去噪 RGB 全细节提取、梯度重建与观测回投",
        transform=ax.transAxes,
        fontsize=20,
        weight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.02,
        0.91,
        "单一 visual_full_detail 产品；不启用 V8 条件门控，也不筛除 RGB 独有纹理、阴影、反光或标记",
        transform=ax.transAxes,
        fontsize=11,
        color=DARK_GRAY,
        va="top",
    )

    x_positions = (0.025, 0.205, 0.385, 0.565, 0.745)
    titles = (
        "① 配准残差前置控制",
        "② RGB 去噪与四尺度分解",
        "③ 最强梯度 + Poisson 重建",
        "④ 全细节注入 NIR/SWIR",
        "⑤ 观测回投与有限锐化",
    )
    bodies = (
        "坐标约定统一\nECC + ROI 仿射/推扫几何\n双向亚像素同名点与边界拒绝",
        "双边保边降噪 0.55\nσ = 0.65/1.35/2.80/5.60 px\n亮度细节 + 0.35 色度细节",
        "逐像素选择灰度/彩色最强梯度\nscreened-Poisson（screen=0.24）\n暗区相对对比增强 0.25",
        "log 乘性增益 0.92\n加性细节 0.22\n允许 RGB 独有结构进入最终产品",
        "增益 0.52–1.92\n4 次最终产品低分辨率回投\nMTF-aware / halo-limited sharpen 0.25",
    )
    faces = (PALE_BLUE, PALE_ORANGE, PALE_ORANGE, PALE_BLUE, PALE_GREEN)
    edges = (BLUE, ORANGE, ORANGE, BLUE, GREEN)
    for x, title, body, face, edge in zip(
        x_positions, titles, bodies, faces, edges, strict=True
    ):
        add_box(
            ax,
            (x, 0.55),
            0.155,
            0.25,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
        )
    for left, right in zip(x_positions[:-1], x_positions[1:], strict=True):
        add_arrow(ax, (left + 0.155, 0.675), (right, 0.675))

    add_box(
        ax,
        (0.06, 0.20),
        0.25,
        0.20,
        "空间目标",
        "裂隙、颗粒、层理、托盘与标签细节\n在 V6 基础上显著增强并向 RGB 视觉清晰度靠近",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=ORANGE,
        body_size=9.1,
    )
    add_box(
        ax,
        (0.375, 0.20),
        0.25,
        0.20,
        "光谱保留范围",
        "最终 HR 产品经 PSF 降采样后回到原 NIR/SWIR 观测\n报告 RMSE、SAM 与波段相关，不改写原尺度光谱曲线",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        body_size=9.0,
    )
    add_box(
        ax,
        (0.69, 0.20),
        0.25,
        0.20,
        "明确的科学边界",
        "新增 HR 像素是 RGB-textured NIR/SWIR 估计\n无独立 HR-HSI 真值时，不声明真实 SWIR 高频无损恢复",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=9.0,
    )
    add_arrow(ax, (0.31, 0.30), (0.375, 0.30), color=ORANGE)
    add_arrow(ax, (0.625, 0.30), (0.69, 0.30), color=GREEN)
    ax.text(
        0.02,
        0.07,
        "核心创新不是增加网络深度，而是把全细节提取、彩色等亮边缘、暗区相对对比、可重建梯度场与传感器观测闭环组合成可审计链条。",
        transform=ax.transAxes,
        color=NAVY,
        fontsize=11,
        weight="bold",
    )
    return save_figure(fig, output_dir, "01_v61_visual_full_detail_route")


def _crop(array: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = xywh
    return array[y : y + height, x : x + width]


def scene_comparison(
    scene: str, scene_data: dict[str, Any], output_dir: Path, number: int
) -> list[str]:
    rgb = scene_data["rgb"]
    v6_native = scene_data["methods"]["V6"]["image"]
    v61_native = scene_data["methods"]["V6.1"]["image"]
    bounds = scene_data["display_bounds"]
    v6 = base.apply_bounds(v6_native, bounds)
    v61 = base.apply_bounds(v61_native, bounds)
    difference = np.linalg.norm(v61_native - v6_native, axis=2)
    difference_scale = max(float(np.percentile(difference, 99.5)), 1e-6)
    difference_display = np.clip(difference / difference_scale, 0.0, 1.0)
    crops = (scene_data["detail_crop"], scene_data["dark_crop"])

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15.8, 13.4),
        gridspec_kw={"height_ratios": [2.35, 1.0, 1.0]},
    )
    titles = (
        "RGB 参考（原色）",
        "V6（共享显示域）",
        "V6.1（共享显示域）",
        "|V6.1 − V6| 反射率差异",
    )
    full_images = (rgb, v6, v61, difference_display)
    for column, (title, image) in enumerate(zip(titles, full_images, strict=True)):
        axes[0, column].imshow(image, cmap="magma" if column == 3 else None)
        axes[0, column].set_title(title, color=NAVY)
        axes[0, column].axis("off")
        if column < 3:
            for crop_index, (x, y, width, height) in enumerate(crops):
                axes[0, column].add_patch(
                    Rectangle(
                        (x, y),
                        width,
                        height,
                        fill=False,
                        linewidth=2.0,
                        edgecolor=ORANGE if crop_index == 0 else SKY,
                    )
                )

    row_names = ("纹理区放大", "暗区放大")
    for row, (row_name, crop_box) in enumerate(zip(row_names, crops, strict=True), start=1):
        crop_images = (
            _crop(rgb, crop_box),
            _crop(v6, crop_box),
            _crop(v61, crop_box),
            _crop(difference_display, crop_box),
        )
        for column, image in enumerate(crop_images):
            axes[row, column].imshow(image, cmap="magma" if column == 3 else None)
            axes[row, column].axis("off")
        axes[row, 0].text(
            -0.06,
            0.5,
            row_name,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            color=ORANGE if row == 1 else SKY,
            weight="bold",
        )

    label = SCENES[scene]["label"]
    v6_metrics = scene_data["methods"]["V6"]["metrics"]
    v61_metrics = scene_data["methods"]["V6.1"]["metrics"]
    fig.suptitle(
        f"{label}：V6 → V6.1 全细节融合（同一反射率显示域）",
        fontsize=18,
        color=NAVY,
        weight="bold",
        y=0.997,
    )
    fig.text(
        0.5,
        0.008,
        (
            f"2201 nm 可靠细节 ρ：{v6_metrics['bands']['2201.0nm']['reliable_rgb_detail']['rho']:.3f} → "
            f"{v61_metrics['bands']['2201.0nm']['reliable_rgb_detail']['rho']:.3f}；"
            f"最终产品回投 RMSE：{v6_metrics['forward_rmse']:.5f} → {v61_metrics['forward_rmse']:.5f}。"
            f"差异图以场景 P99.5={difference_scale:.4f} 反射率归一化。"
        ),
        ha="center",
        fontsize=9.5,
        color=DARK_GRAY,
    )
    fig.subplots_adjust(top=0.956, bottom=0.042, wspace=0.035, hspace=0.055)
    return save_figure(
        fig, output_dir, f"{number:02d}_{scene}_v6_v61_shared_domain_comparison"
    )


def key_metric_figure(data: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.9))
    scene_order = ("3dssz", "zkh3")
    labels = [SCENES[scene]["label"] for scene in scene_order]
    x = np.arange(len(scene_order))
    width = 0.34

    panels = [
        (
            "boundary_correlation",
            "RGB–材料边界相关（高为优）",
            lambda metric: metric["boundary_correlation"],
            (0.0, 0.55),
        ),
        (
            "rho_2201",
            "2201 nm 可靠细节 ρ（高为优）",
            lambda metric: metric["bands"]["2201.0nm"]["reliable_rgb_detail"]["rho"],
            (0.0, 0.85),
        ),
        (
            "dark_rho_2201",
            "2201 nm 暗区可靠纹理 ρ（高为优）",
            lambda metric: metric["bands"]["2201.0nm"]["dark_reliable_rgb_detail"]["rho"],
            (0.0, 0.90),
        ),
        (
            "forward_rmse",
            "最终产品回投 RMSE（低为优）",
            lambda metric: metric["forward_rmse"],
            (0.0, 0.017),
        ),
        (
            "forward_sam",
            "最终产品回投 SAM / °（低为优）",
            lambda metric: metric["forward_sam_deg"],
            (0.0, 1.10),
        ),
        (
            "edge_f1",
            "2201 nm 1 px edge F1（高为优）",
            lambda metric: metric["bands"]["2201.0nm"]["edge_f1_1px"],
            (0.0, 0.82),
        ),
    ]
    for ax, (_, title, getter, ylim) in zip(axes.flat, panels, strict=True):
        for method_index, method in enumerate(METHODS):
            values = [
                getter(data[scene]["methods"][method]["metrics"])
                for scene in scene_order
            ]
            positions = x + (method_index - 0.5) * width
            bars = ax.bar(
                positions,
                values,
                width,
                color=METHOD_COLORS[method],
                label=method,
            )
            for bar, value in zip(bars, values, strict=True):
                decimals = 5 if "RMSE" in title else 3
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim[1] - ylim[0]) * 0.025,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=NAVY,
                )
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "V6.1 在两组真实 ROI 上同时增强空间细节并改善最终产品观测回投",
        color=NAVY,
        fontsize=17,
        weight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.015,
        "ρ、edge F1 与边界相关是同数据 RGB 引导诊断；RMSE/SAM 是原始尺度观测一致性，不是独立 HR-SWIR 真值。",
        ha="center",
        color=VERMILION,
        fontsize=9.5,
        weight="bold",
    )
    fig.subplots_adjust(top=0.89, bottom=0.10, hspace=0.39, wspace=0.25)
    return save_figure(fig, output_dir, "04_v6_v61_key_metric_improvement")


def detail_amplitude_audit(data: dict[str, Any], output_dir: Path) -> list[str]:
    categories = [
        (scene, wavelength)
        for scene in ("3dssz", "zkh3")
        for wavelength in ("901.0nm", "1651.0nm", "2201.0nm")
    ]
    labels = [
        f"{SCENES[scene]['label']}\n{wavelength.replace('.0nm', ' nm')}"
        for scene, wavelength in categories
    ]
    panels = (
        ("rho", "ρ：相干性（高为优）", (0.0, 0.9), None),
        ("beta", "β：RGB 等效相对对比斜率（目标≈1）", (0.0, 2.0), (0.8, 1.2)),
        ("energy_ratio_A", "A：高频能量比（目标≈1）", (0.0, 3.4), (0.8, 1.25)),
        (
            "orthogonal_residual_ratio_R_perp",
            "R⊥：非 RGB 对齐高频（低为优）",
            (0.0, 3.2),
            (0.0, 0.35),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 9.3))
    x = np.arange(len(categories))
    width = 0.34
    for ax, (metric_name, title, ylim, target) in zip(
        axes.flat, panels, strict=True
    ):
        if target is not None:
            ax.axhspan(target[0], target[1], color=GREEN, alpha=0.10, zorder=0)
        for method_index, method in enumerate(METHODS):
            values = [
                float(
                    data[scene]["methods"][method]["metrics"]["bands"][wavelength][
                        "reliable_rgb_detail"
                    ][metric_name]
                )
                for scene, wavelength in categories
            ]
            ax.bar(
                x + (method_index - 0.5) * width,
                values,
                width,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "V6.1 全细节注入的幅值与残差审计（σ = 2.4 px）",
        color=NAVY,
        fontsize=17,
        weight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.015,
        "V6.1 的目标是视觉全细节，不以保守门控区间作为拒绝条件；A 与 R⊥ 升高应被解释为更强的 RGB 纹理迁移及潜在伪影风险，而不是 SWIR 真值增加。",
        ha="center",
        color=VERMILION,
        fontsize=9.3,
        weight="bold",
    )
    fig.subplots_adjust(top=0.89, bottom=0.12, hspace=0.39, wspace=0.22)
    return save_figure(fig, output_dir, "05_v61_detail_amplitude_and_residual_audit")


def _predict_low_cube(
    repo: Path, run_dir: Path, api: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from geocorefusion.degradation import PsfModel
    from geocorefusion.quality import _degrade_final_modulated_cube

    processing = read_json(run_dir / "metadata" / "processing_config.json")
    fusion_config = fusion_config_from_payload(
        processing["fusion"], api["FusionConfig"]
    )
    low_mmap, low_meta = api["open_cube"](
        run_dir / "analysis" / "harmonized_lowres.hdr"
    )
    observed = np.asarray(low_mmap, dtype=np.float32).copy()
    model, _ = api["fit_subspace"](
        observed,
        rank=int(fusion_config.rank),
        max_pixels=int(fusion_config.max_basis_pixels),
        random_seed=int(fusion_config.random_seed),
        clip_quantiles=tuple(fusion_config.clip_quantiles),
    )
    coeff_mmap, _ = api["open_cube"](
        run_dir / "coefficients" / "material_coefficients.hdr"
    )
    coefficients = np.asarray(coeff_mmap, dtype=np.float32)
    gain = base.load_scalar_map(run_dir / "metrics" / "spatial_detail_gain.hdr", api)
    additive = base.load_scalar_map(
        run_dir / "metrics" / "spatial_additive_detail.hdr", api
    )
    additive_scale = api["build_additive_spectral_scale"](
        observed, fusion_config
    )
    psf_payload = read_json(run_dir / "metadata" / "psf_model.json")
    psf = PsfModel(
        sigma_x_highres=float(psf_payload["sigma_x_highres"]),
        sigma_y_highres=float(psf_payload["sigma_y_highres"]),
        score=float(psf_payload["score"]),
        low_shape=tuple(int(v) for v in psf_payload["low_shape"]),
        high_shape=tuple(int(v) for v in psf_payload["high_shape"]),
        method=str(psf_payload.get("method", "anisotropic_gaussian_grid_search")),
    )
    predicted, _ = _degrade_final_modulated_cube(
        coefficients,
        model,
        psf,
        gain,
        additive,
        additive_scale,
        physical_clip_limits=(0.0, None),
        chunk_bands=8,
    )
    return observed, predicted, np.asarray(low_meta.wavelengths, dtype=np.float32)


def _select_spectral_pixels(cube: np.ndarray) -> list[tuple[int, int, str]]:
    mean = np.nanmean(cube, axis=2)
    variability = np.nanstd(cube, axis=2)
    valid = np.isfinite(mean) & np.isfinite(variability)
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    labels = ("暗反射率", "中等反射率", "亮反射率")
    percentiles = (18.0, 50.0, 82.0)
    selected: list[tuple[int, int, str]] = []
    for percentile, label in zip(percentiles, labels, strict=True):
        target = float(np.percentile(mean[valid], percentile))
        tolerance = max(float(np.std(mean[valid])) * 0.08, 1e-5)
        candidates = valid & (np.abs(mean - target) <= tolerance)
        if not np.any(candidates):
            candidates = valid
        score = np.where(candidates, variability - 0.2 * np.abs(mean - target), -np.inf)
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        selected.append((int(y), int(x), label))
    return selected


def spectral_curve_figure(
    repo: Path, data: dict[str, Any], api: dict[str, Any], output_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    fig, axes = plt.subplots(2, 3, figsize=(16.3, 8.7), sharex=True)
    summary: dict[str, Any] = {}
    for row, scene in enumerate(("3dssz", "zkh3")):
        run_dir = data[scene]["methods"]["V6.1"]["run_dir"]
        observed, predicted, wavelengths = _predict_low_cube(repo, run_dir, api)
        selected = _select_spectral_pixels(observed)
        scene_samples: list[dict[str, Any]] = []
        for column, (y, x, label) in enumerate(selected):
            ax = axes[row, column]
            observation = observed[y, x]
            reprojection = predicted[y, x]
            valid = np.isfinite(observation) & np.isfinite(reprojection)
            ax.plot(
                wavelengths[valid],
                observation[valid],
                color=BLUE,
                linewidth=1.8,
                label="原始尺度观测",
            )
            ax.plot(
                wavelengths[valid],
                reprojection[valid],
                color=ORANGE,
                linewidth=1.35,
                linestyle="--",
                label="V6.1 HR→PSF 回投",
            )
            rmse_value = float(
                np.sqrt(np.mean((observation[valid] - reprojection[valid]) ** 2))
            )
            denominator = np.linalg.norm(observation[valid]) * np.linalg.norm(
                reprojection[valid]
            )
            cosine = float(
                np.clip(
                    np.dot(observation[valid], reprojection[valid])
                    / max(float(denominator), 1e-12),
                    -1.0,
                    1.0,
                )
            )
            sam = float(np.degrees(np.arccos(cosine)))
            ax.set_title(
                f"{SCENES[scene]['label']} · {label}\nLR({x},{y})  RMSE={rmse_value:.4f}, SAM={sam:.2f}°"
            )
            ax.grid(color=GRID, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("波长 / nm")
            if column == 0:
                ax.set_ylabel("归一化反射率")
            scene_samples.append(
                {
                    "label": label,
                    "lowres_xy": [x, y],
                    "rmse": rmse_value,
                    "sam_deg": sam,
                }
            )
        summary[scene] = {
            "samples": scene_samples,
            "global_metrics": {
                key: data[scene]["methods"]["V6.1"]["metrics"][key]
                for key in ("forward_rmse", "forward_sam_deg", "forward_band_cc")
            },
        }
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "V6.1 光谱曲线仍在：最终 HR 产品回投到原传感器网格后与观测重合",
        color=NAVY,
        fontsize=17,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "曲线重合证明原始尺度光谱观测一致性；它不证明每个新增高分辨率像素都具有独立真实的 SWIR 光谱。",
        ha="center",
        color=VERMILION,
        fontsize=9.5,
        weight="bold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, hspace=0.39, wspace=0.22)
    return (
        save_figure(fig, output_dir, "06_v61_low_resolution_spectral_curve_consistency"),
        summary,
    )


def registration_control_figure(
    repo: Path, data: dict[str, Any], output_dir: Path
) -> list[str]:
    benchmark_path = repo / "artifacts" / "v6_research" / "experiments" / "benchmark_summary.json"
    benchmark = read_json(benchmark_path)
    synthetic = benchmark["registration_synthetic"]

    fig = plt.figure(figsize=(17.2, 8.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)
    route_ax = fig.add_subplot(grid[0, 0])
    route_ax.axis("off")
    route_ax.text(
        0.02,
        0.94,
        "配准残差在细节注入前被显式处理",
        transform=route_ax.transAxes,
        fontsize=16,
        color=NAVY,
        weight="bold",
    )
    route_ax.text(
        0.02,
        0.895,
        "先控制几何，再允许全 RGB 细节进入 NIR/SWIR",
        transform=route_ax.transAxes,
        fontsize=10.5,
        color=DARK_GRAY,
    )
    stages = (
        ("统一坐标与有效边界", "像素中心约定\nNaN/常量边界拒绝", PALE_BLUE, BLUE),
        ("全幅结构粗配准", "跨模态结构特征\nphase + ECC / affine", PALE_BLUE, BLUE),
        ("ROI 推扫几何", "列向低自由度修正\n三模态联合一致性", PALE_ORANGE, ORANGE),
        ("双向亚像素同名点", "峰值亚像素拟合\n前后向误差 + margin", PALE_ORANGE, ORANGE),
        ("一次原始立方体重采样", "避免反复插值\n通过后才提取 RGB 细节", PALE_GREEN, GREEN),
    )
    y_positions = (0.72, 0.565, 0.41, 0.255, 0.10)
    for y, (title, body, face, edge) in zip(y_positions, stages, strict=True):
        add_box(
            route_ax,
            (0.10, y),
            0.70,
            0.115,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
            title_size=10.5,
            body_size=8.7,
        )
    for start, end in zip(y_positions[:-1], y_positions[1:], strict=True):
        add_arrow(
            route_ax,
            (0.45, start),
            (0.45, end + 0.115),
            color=BLUE,
        )

    chart_grid = grid[0, 1].subgridspec(2, 1, hspace=0.38)
    score_ax = fig.add_subplot(chart_grid[0, 0])
    labels: list[str] = []
    before: list[float] = []
    after: list[float] = []
    for scene in ("3dssz", "zkh3"):
        registration = data[scene]["methods"]["V6.1"]["metrics"]["registration"]
        for sensor in ("nir", "swir"):
            labels.append(f"{SCENES[scene]['label']}\n{sensor.upper()}")
            before.append(registration[sensor]["score_before"])
            after.append(registration[sensor]["score_after"])
    x = np.arange(len(labels))
    width = 0.34
    score_ax.bar(x - width / 2, before, width, color=GRAY, label="ROI 修正前")
    score_ax.bar(x + width / 2, after, width, color=BLUE, label="ROI 修正后")
    score_ax.set_xticks(x, labels)
    score_ax.set_ylim(0.0, 0.86)
    score_ax.set_ylabel("跨模态结构得分")
    score_ax.set_title("真实 ROI：配准结构得分提升")
    score_ax.grid(axis="y", color=GRID, linewidth=0.8)
    score_ax.legend(frameon=False, ncol=2, loc="upper left")
    score_ax.spines[["top", "right"]].set_visible(False)

    tre_ax = fig.add_subplot(chart_grid[1, 0])
    names = ("全幅粗仿射 TRE", "ROI 仿射 TRE", "局部场 EPE")
    entries = (
        synthetic["coarse_affine_tre"],
        synthetic["roi_affine_tre"],
        synthetic["dense_residual_epe"],
    )
    median = np.asarray([entry["median_px"] for entry in entries])
    p95 = np.asarray([entry["p95_px"] for entry in entries])
    y = np.arange(len(names))
    tre_ax.barh(y + 0.17, p95, height=0.30, color=BLUE, label="P95")
    tre_ax.barh(y - 0.17, median, height=0.30, color=GREEN, label="Median")
    tre_ax.axvline(0.5, color=ORANGE, linestyle="--", linewidth=1.5)
    tre_ax.set_yticks(y, names)
    tre_ax.invert_yaxis()
    tre_ax.set_xlabel("分析网格像素")
    tre_ax.set_title("受控合成真值：组件级亚像素证据")
    tre_ax.grid(axis="x", color=GRID, linewidth=0.8)
    tre_ax.legend(frameon=False, ncol=2, loc="lower right")
    tre_ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(p95):
        tre_ax.text(value + 0.008, index + 0.17, f"{value:.3f}", va="center", fontsize=8.5)

    fig.suptitle(
        "V6.1 延续最新配准前端：支持亚像素估计，但真实岩心精度仍需独立靶标 TRE",
        color=NAVY,
        fontsize=17,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "真实 ROI 的结构得分不能替代独立同名点误差；当前可严谨声明的是算法组件具备亚像素能力、生产数据配准通过内部结构审计。",
        ha="center",
        color=VERMILION,
        fontsize=9.3,
        weight="bold",
    )
    fig.subplots_adjust(top=0.91, bottom=0.08)
    return save_figure(fig, output_dir, "07_v61_registration_residual_control")


def product_contract_figure(output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(17.2, 7.7))
    ax.axis("off")
    ax.text(
        0.02,
        0.94,
        "V6.1 单产品合同：空间全细节与原尺度光谱一致性同时保留",
        transform=ax.transAxes,
        fontsize=19,
        color=NAVY,
        weight="bold",
    )
    add_box(
        ax,
        (0.04, 0.50),
        0.21,
        0.25,
        "原始观测",
        "RGB（高空间）\nNIR/SWIR（低空间、真实光谱）",
        face=PALE_BLUE,
        edge=BLUE,
        title_color=BLUE,
        body_size=9.5,
    )
    add_box(
        ax,
        (0.315, 0.50),
        0.25,
        0.25,
        "V6.1 融合算子",
        "去噪 RGB 全细节梯度场\nlog 增益 + 加性细节\n低分辨率观测回投",
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=ORANGE,
        body_size=9.4,
    )
    add_box(
        ax,
        (0.63, 0.50),
        0.31,
        0.25,
        "唯一交付产品",
        "geocorefusion.visual-full-detail.v1\n高空间 RGB-textured NIR/SWIR 估计\n裂隙、颗粒、暗区和标记均允许进入",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        body_size=9.2,
    )
    add_arrow(ax, (0.25, 0.625), (0.315, 0.625), color=BLUE)
    add_arrow(ax, (0.565, 0.625), (0.63, 0.625), color=ORANGE)
    add_box(
        ax,
        (0.12, 0.15),
        0.31,
        0.19,
        "可以验证并报告",
        "共享显示域空间改善；暗区/边缘/幅值诊断\n最终产品回投 RMSE、SAM、band-CC",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        body_size=9.2,
    )
    add_box(
        ax,
        (0.57, 0.15),
        0.31,
        0.19,
        "不能由当前数据单独证明",
        "每个新增 HR 像素的真实 SWIR 光谱\nRGB 独有纹理对应真实 SWIR 高频结构",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=9.2,
    )
    ax.text(
        0.5,
        0.055,
        "这是单一视觉全细节产品，不保留 scientific_conditional/V8 门控分支。",
        transform=ax.transAxes,
        ha="center",
        color=NAVY,
        fontsize=11,
        weight="bold",
    )
    return save_figure(fig, output_dir, "08_v61_single_product_and_claim_contract")


def serializable_summary(data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for scene, scene_data in data.items():
        payload[scene] = {
            "label": SCENES[scene]["label"],
            "display_bounds": [list(map(float, pair)) for pair in scene_data["display_bounds"]],
            "detail_crop_xywh": list(scene_data["detail_crop"]),
            "dark_crop_xywh": list(scene_data["dark_crop"]),
            "methods": {},
        }
        for method in METHODS:
            method_data = scene_data["methods"][method]
            payload[scene]["methods"][method] = {
                "run": method_data["run"],
                "metrics": method_data["metrics"],
                "reconstruction_diagnostics": method_data["diagnostics"],
            }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for figures"
    )
    parser.add_argument(
        "--summary", type=Path, required=True, help="Benchmark summary JSON"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260720)
    setup_style()
    api = base.import_project(repo)
    data = collect_data(repo, api)

    files: list[str] = []
    files += method_route(output_dir)
    files += scene_comparison("3dssz", data["3dssz"], output_dir, 2)
    files += scene_comparison("zkh3", data["zkh3"], output_dir, 3)
    files += key_metric_figure(data, output_dir)
    files += detail_amplitude_audit(data, output_dir)
    spectral_files, spectral_summary = spectral_curve_figure(
        repo, data, api, output_dir
    )
    files += spectral_files
    files += registration_control_figure(repo, data, output_dir)
    files += product_contract_figure(output_dir)

    summary = {
        "schema": "geocorefusion.v61-visual-benchmark.v1",
        "method": "V6.1 visual_full_detail",
        "scientific_conditional_included": False,
        "comparison_domain": (
            "V6 and V6.1 reconstructed reflectance with one shared per-scene linear display mapping"
        ),
        "truth_scope": (
            "same-data RGB structural-transfer diagnostics plus low-resolution observation consistency; no independent HR-SWIR truth"
        ),
        "scenes": serializable_summary(data),
        "spectral_curve_samples": spectral_summary,
    }
    args.summary.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "schema": "geocorefusion.v61-figure-manifest.v1",
        "figures": files,
        "benchmark_summary": str(args.summary),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
