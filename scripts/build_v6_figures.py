"""Build publication-ready V6 method, evidence, and experiment figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


NAVY = "#17365D"
BLUE = "#2E74B5"
PALE_BLUE = "#DCEAF7"
ORANGE = "#E67E22"
PALE_ORANGE = "#FBE5D6"
GREEN = "#2F8F72"
PALE_GREEN = "#DDEFE8"
GRAY = "#5F6978"
LIGHT_GRAY = "#EFF2F5"
RED = "#C94B4B"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_box(ax, xy, width, height, title, body, *, face=PALE_BLUE, edge=BLUE, title_color=NAVY):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        weight="bold",
        color=title_color,
        transform=ax.transAxes,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=8.6,
        transform=ax.transAxes,
        linespacing=1.35,
    )
    return box


def arrow(ax, start, end, *, color=BLUE, width=1.7):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def registration_route(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.axis("off")
    ax.text(0.02, 0.95, "V6 配准：三模态闭环、坐标一致与可验证精度", color=NAVY, fontsize=20, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.905, "Coordinate-consistent, cycle-constrained low-DOF push-broom registration", color=GRAY, fontsize=11, transform=ax.transAxes)
    xs = [0.025, 0.18, 0.335, 0.49, 0.645, 0.80]
    titles = [
        "异构输入",
        "端点坐标统一",
        "全局粗配准",
        "稳健局部点",
        "低自由度形变",
        "闭环验收",
    ]
    bodies = [
        "RGB 面阵\nNIR / SWIR 推扫",
        "(N−1)/(n−1)\n同一像素中心约定",
        "NGF-like 结构域\nphase + ECC / affine",
        "双向一致 + 峰裕量\n边界峰拒绝 + MAD",
        "置信加权平滑场\nJacobian / 支持域门控",
        "RGB / NIR / SWIR\n一次原始立方体重采样",
    ]
    for x, title, body in zip(xs, titles, bodies, strict=True):
        add_box(ax, (x, 0.53), 0.135, 0.23, title, body)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + 0.135, 0.645), (xs[i + 1], 0.645))

    add_box(ax, (0.07, 0.18), 0.23, 0.20, "合成真值层", "已知仿射 / 平滑场\nTRE、EPE、P95、Jacobian", face=PALE_GREEN, edge=GREEN, title_color=GREEN)
    add_box(ax, (0.385, 0.18), 0.23, 0.20, "真实盲测层", "未参与拟合的人工地标\nHSI px / RGB px / mm + 95% CI", face=PALE_ORANGE, edge=ORANGE, title_color=ORANGE)
    add_box(ax, (0.70, 0.18), 0.23, 0.20, "论文声明边界", "合成可声明“亚像素”\n真实数据未有 TRE 前不得声明", face=LIGHT_GRAY, edge=GRAY, title_color=NAVY)
    arrow(ax, (0.30, 0.28), (0.385, 0.28), color=GREEN)
    arrow(ax, (0.615, 0.28), (0.70, 0.28), color=ORANGE)
    ax.text(0.02, 0.06, "核心创新：不是堆叠网络，而是把坐标约定、三模态几何闭环、局部置信与精度证据放在同一可复现框架中。", color=NAVY, fontsize=11, weight="bold", transform=ax.transAxes)
    fig.savefig(out / "01_registration_v6_route.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fusion_route(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.axis("off")
    ax.text(0.02, 0.95, "V6 融合：SAMI-HPM 阴影稳健材料自适应高通调制", color=NAVY, fontsize=20, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.905, "Shadow-Aware Material-Adaptive Intensity High-Pass Modulation with Observation Consistency", color=GRAY, fontsize=11, transform=ax.transAxes)

    add_box(ax, (0.025, 0.58), 0.16, 0.23, "RGB 结构支路", "Y = 0.299R+0.587G+0.114B\nlog 相对对比度，分离照明/反射", face=PALE_ORANGE, edge=ORANGE, title_color=ORANGE)
    add_box(ax, (0.025, 0.26), 0.16, 0.23, "NIR/SWIR 光谱支路", "重叠区光谱统一\nXLR = μ + ALR E", face=PALE_GREEN, edge=GREEN, title_color=GREEN)
    add_box(ax, (0.245, 0.58), 0.17, 0.23, "多尺度内在细节", "d0 = [l-G1.2(l)] + 0.55[G1.2(l)-G7.5(l)]\n暗部使用相对对比而非绝对亮度", face=PALE_ORANGE, edge=ORANGE, title_color=ORANGE)
    add_box(ax, (0.245, 0.26), 0.17, 0.23, "材料系数恢复", "有符号岭回归 + 材料支持\nRGB 不改写光谱基 E", face=PALE_GREEN, edge=GREEN, title_color=GREEN)
    add_box(ax, (0.475, 0.43), 0.17, 0.25, "观测近零空间", "d ← d − U[D(d)]\n系数细节同样投影\n抑制低频观测漂移", face=PALE_BLUE, edge=BLUE, title_color=NAVY)
    add_box(ax, (0.705, 0.58), 0.17, 0.23, "指数型同谱增益", "g = exp(αd)\n同像元所有波段乘同一正数\n裁剪前光谱角不变", face=PALE_BLUE, edge=BLUE, title_color=NAVY)
    add_box(ax, (0.705, 0.26), 0.17, 0.23, "传感器回投影", "g ← g − U[D(g)−1]\nALR 残差回投影\n保持 NIR/SWIR 自一致", face=PALE_BLUE, edge=BLUE, title_color=NAVY)
    add_box(ax, (0.91, 0.43), 0.075, 0.25, "XHR", "(μ+AHR E)·g\n细节 + 光谱\n+ 不确定度", face=LIGHT_GRAY, edge=NAVY, title_color=NAVY)
    arrow(ax, (0.185, 0.695), (0.245, 0.695), color=ORANGE)
    arrow(ax, (0.185, 0.375), (0.245, 0.375), color=GREEN)
    arrow(ax, (0.415, 0.695), (0.475, 0.575), color=ORANGE)
    arrow(ax, (0.415, 0.375), (0.475, 0.525), color=GREEN)
    arrow(ax, (0.645, 0.575), (0.705, 0.695), color=BLUE)
    arrow(ax, (0.645, 0.525), (0.705, 0.375), color=BLUE)
    arrow(ax, (0.875, 0.695), (0.91, 0.575), color=BLUE)
    arrow(ax, (0.875, 0.375), (0.91, 0.525), color=BLUE)
    ax.text(0.03, 0.10, "目标：尽量恢复 RGB 支持且局部 SNR 可靠区域的几何相对对比度；不声称 RGB 提供 1650/2200/2350 nm 的独立高频真值。", color=NAVY, fontsize=11.5, weight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.055, "V6 默认 α=0.20；加性共享细节关闭，避免暗波段通过加法产生光谱幻觉。", color=GRAY, fontsize=10, transform=ax.transAxes)
    fig.savefig(out / "02_fusion_v6_route.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def registration_accuracy(summary: dict, out: Path) -> None:
    reg = summary["registration_synthetic"]
    names = ["全幅粗仿射 TRE", "ROI 仿射 TRE", "局部场 EPE"]
    entries = [reg["coarse_affine_tre"], reg["roi_affine_tre"], reg["dense_residual_epe"]]
    med = np.asarray([entry["median_px"] for entry in entries])
    p95 = np.asarray([entry["p95_px"] for entry in entries])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.barh(y + 0.17, p95, height=0.30, color=BLUE, label="P95")
    ax.barh(y - 0.17, med, height=0.30, color=GREEN, label="Median")
    ax.axvline(0.5, color=ORANGE, linestyle="--", linewidth=1.6, label="0.5 analysis px")
    for index, value in enumerate(p95):
        ax.text(value + 0.012, index + 0.17, f"{value:.3f}", va="center", color=NAVY, fontsize=9)
    for index, value in enumerate(med):
        ax.text(value + 0.012, index - 0.17, f"{value:.3f}", va="center", color=NAVY, fontsize=9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("误差（分析网格像素）")
    ax.set_title("受控合成真值：算法组件配准误差")
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(0.0, -0.28, "注意：这是受控组件试验而非生产配置复现；真实岩心仍需独立人工地标 TRE。", transform=ax.transAxes, color=RED, fontsize=9.5, weight="bold")
    fig.savefig(out / "03_registration_synthetic_accuracy.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def fusion_metrics(summary: dict, out: Path) -> None:
    real = summary["real_roi_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax = axes[0]
    labels = []
    v5, v6 = [], []
    for scene in ("3dssz", "zkh3"):
        for band in ("901.0nm", "1651.0nm", "2201.0nm"):
            labels.append(f"{scene.upper()}\n{band.replace('.0','')}")
            v5.append(real[scene]["v5_matched"]["dark_log_detail_correlation"][band])
            v6.append(real[scene]["v6"]["dark_log_detail_correlation"][band])
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, v5, width, label="V5 matched", color="#A8B3BF")
    ax.bar(x + width / 2, v6, width, label="V6 SAMI-HPM", color=BLUE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("暗部 20% log 高频相关")
    ax.set_title("真实 ROI：暗部几何细节传递")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.legend(frameon=False, loc="upper center", ncol=2)

    ax2 = axes[1]
    scenes = ["3DSSZ", "ZKH3"]
    indices = np.arange(2)
    rmse_v5 = [real[s.lower()]["v5_matched"]["continuous_cube_rmse"] for s in scenes]
    rmse_v6 = [real[s.lower()]["v6"]["continuous_cube_rmse"] for s in scenes]
    ax2.bar(indices - width / 2, rmse_v5, width, color="#A8B3BF", label="V5 matched")
    ax2.bar(indices + width / 2, rmse_v6, width, color=GREEN, label="V6")
    ax2.set_xticks(indices, scenes)
    ax2.set_ylabel("低分辨率连续立方体 RMSE")
    ax2.set_title("观测一致性（越低越好）")
    ax2.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax2.ticklabel_format(axis="y", style="plain")
    for index, value in enumerate(rmse_v6):
        ax2.text(index + width / 2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8.5, color=NAVY)
    fig.suptitle("V5→V6：暗部细节显著增强，观测误差保持同量级", color=NAVY, fontsize=16, weight="bold", y=1.02)
    fig.text(0.5, -0.03, "相关性是结构传递诊断，不是独立 SWIR 高频真值。所有 V5/V6 实验使用同一修正后配准几何。", ha="center", color=RED, fontsize=9.5, weight="bold")
    fig.savefig(out / "04_fusion_real_metrics.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def select_dark_texture_crop(rgb: np.ndarray, crop_w: int = 220, crop_h: int = 270) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    best = None
    for y in range(20, max(21, gray.shape[0] - crop_h - 20), 32):
        for x in range(20, max(21, gray.shape[1] - crop_w - 20), 28):
            patch = gray[y : y + crop_h, x : x + crop_w]
            mean = float(np.mean(patch))
            if not 0.025 <= mean <= 0.38:
                continue
            texture = float(np.percentile(grad[y : y + crop_h, x : x + crop_w], 85))
            darkness = max(0.0, 0.42 - mean)
            score = texture * (0.5 + darkness)
            if best is None or score > best[0]:
                best = (score, x, y)
    if best is None:
        return (gray.shape[1] // 4, gray.shape[0] // 3, crop_w, crop_h)
    return (best[1], best[2], crop_w, crop_h)


def comparison_figure(repo: Path, scene: str, summary: dict, out: Path) -> None:
    base = repo / "runs"
    rgb = np.asarray(Image.open(base / f"{scene}_roi_fusion_v6" / "previews" / "rgb_reference.png").convert("RGB"))
    v5 = np.asarray(Image.open(base / f"{scene}_roi_fusion_v5_matched" / "previews" / "fused_false_color_2200_1650_900.png").convert("RGB"))
    v6 = np.asarray(Image.open(base / f"{scene}_roi_fusion_v6" / "previews" / "fused_false_color_2200_1650_900.png").convert("RGB"))
    crop = select_dark_texture_crop(rgb)
    x, y, w, h = crop
    images = [rgb, v5, v6]
    titles = ["RGB reference", "V5 matched", "V6 SAMI-HPM"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6))
    for col, (image, title) in enumerate(zip(images, titles, strict=True)):
        axes[0, col].imshow(image)
        axes[0, col].add_patch(Rectangle((x, y), w, h, edgecolor=ORANGE, facecolor="none", linewidth=2.0))
        axes[0, col].set_title(title, color=NAVY)
        axes[0, col].axis("off")
        axes[1, col].imshow(image[y : y + h, x : x + w])
        axes[1, col].axis("off")
    axes[1, 0].set_title("暗部低对比 ROI", color=ORANGE)
    metrics = summary["real_roi_matched"][scene]
    v5_dark = np.mean(list(metrics["v5_matched"]["dark_log_detail_correlation"].values()))
    v6_dark = np.mean(list(metrics["v6"]["dark_log_detail_correlation"].values()))
    fig.suptitle(f"{scene.upper()}：V5 与 V6 真实 ROI 同几何对比", fontsize=16, weight="bold", color=NAVY)
    fig.text(0.5, 0.025, f"暗部三波段平均 log-HF 相关：V5 {v5_dark:.3f} → V6 {v6_dark:.3f}；各预览独立 2–98% 拉伸，仅比较结构可见性，不比较绝对对比度或辐射保真。", ha="center", color=GRAY, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure_number = "05" if scene.lower() == "3dssz" else "06"
    fig.savefig(out / f"{figure_number}_{scene}_v5_v6_visual_comparison.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def classic_vs_v6(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.axis("off")
    ax.text(0.03, 0.94, "为什么经典锐化结果看起来“几乎就是 RGB/PAN”？", fontsize=19, weight="bold", color=NAVY, transform=ax.transAxes)
    add_box(ax, (0.04, 0.56), 0.25, 0.24, "经典强度替换/调制", "Gram–Schmidt / PC / HSV\nBrovey / NNDiffuse\n直接把高分辨率强度或高频写入输出", face=PALE_ORANGE, edge=ORANGE, title_color=ORANGE)
    add_box(ax, (0.375, 0.56), 0.25, 0.24, "视觉结果", "边缘、MTF 与 PAN/RGB 非常接近\n自然显示尺度下肉眼差距很小", face=PALE_ORANGE, edge=ORANGE, title_color=ORANGE)
    add_box(ax, (0.71, 0.56), 0.25, 0.24, "隐含前提与风险", "PAN 与目标波段共享谱域响应\n跨到 1650/2200 nm 时可能复制伪纹理", face=LIGHT_GRAY, edge=RED, title_color=RED)
    arrow(ax, (0.29, 0.68), (0.375, 0.68), color=ORANGE)
    arrow(ax, (0.625, 0.68), (0.71, 0.68), color=RED)

    add_box(ax, (0.04, 0.18), 0.25, 0.24, "GeoCoreFusion V6", "RGB 只提供结构\nlog 相对对比 + 材料系数 + 同谱指数增益", face=PALE_BLUE, edge=BLUE, title_color=NAVY)
    add_box(ax, (0.375, 0.18), 0.25, 0.24, "物理约束", "近零空间投影 + 回投影\nNIR/SWIR 重叠区 + 不确定度", face=PALE_GREEN, edge=GREEN, title_color=GREEN)
    add_box(ax, (0.71, 0.18), 0.25, 0.24, "正确目标", "几何细节尽量接近 RGB\n光谱颜色无需、也不应等同 RGB", face=PALE_BLUE, edge=BLUE, title_color=NAVY)
    arrow(ax, (0.29, 0.30), (0.375, 0.30), color=BLUE)
    arrow(ax, (0.625, 0.30), (0.71, 0.30), color=GREEN)
    ax.text(0.03, 0.07, "结论：经典方法的“肉眼无差别”是空间强度继承的结果，不等价于每个 SWIR 波段拥有真实 RGB 高频。", color=RED, fontsize=11.5, weight="bold", transform=ax.transAxes)
    fig.savefig(out / "07_classical_pansharpening_vs_v6.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def evidence_ladder(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.axis("off")
    ax.text(0.05, 0.94, "证据阶梯：从“更清楚”到“可发表的科学结论”", fontsize=19, weight="bold", color=NAVY, transform=ax.transAxes)
    levels = [
        (0.08, 0.13, 0.72, 0.14, LIGHT_GRAY, GRAY, "L1 视觉与自一致", "棋盘图、暗部细节、降质 RMSE/SAM —— 当前两景真实 ROI 已具备"),
        (0.14, 0.32, 0.66, 0.14, PALE_BLUE, BLUE, "L2 合成真值", "已知仿射/形变 TRE、EPE、Wald HR-HSI —— 配准已具备；融合需扩展公共数据"),
        (0.20, 0.51, 0.60, 0.14, PALE_ORANGE, ORANGE, "L3 独立真实真值", "人工地标、重复扫描、HR-NIR/SWIR、点光谱/XRD/Raman —— 当前缺失"),
        (0.26, 0.70, 0.54, 0.14, PALE_GREEN, GREEN, "L4 跨钻孔与地质效用", "3+ 钻孔、30–50 箱、冻结参数、矿物/岩性/边界下游验证 —— 投稿主证据"),
    ]
    for x, y, w, h, face, edge, title, body in levels:
        add_box(ax, (x, y), w, h, title, body, face=face, edge=edge, title_color=edge if edge != GRAY else NAVY)
    ax.text(0.05, 0.055, "当前可诚实结论：V6 已在合成配准与两景真实暗部诊断上显示改进；尚不能宣称真实 RGB 网格亚像素或 SWIR 细节“无损真值”。", color=RED, fontsize=10.5, weight="bold", transform=ax.transAxes)
    fig.savefig(out / "08_evidence_ladder.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def experiment_matrix(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.axis("off")
    ax.text(0.025, 0.95, "高水平论文最小实验矩阵", fontsize=20, weight="bold", color=NAVY, transform=ax.transAxes)
    columns = ["配准", "融合", "光谱统一", "地质效用"]
    rows = ["基线", "合成真值", "真实盲测", "核心消融", "主指标", "通过门槛"]
    cells = {
        (0, 0): "全局 ECC / affine\nRIFT/3MRS 或可复现强基线",
        (0, 1): "Bicubic、GS/PC/NNDiffuse\nCNMF、HySure、2022+ 强基线",
        (0, 2): "无校正、全局 gain/offset\n波长相关重叠区校正",
        (0, 3): "原始 HSI、V5、V6\n专家人工编录",
        (1, 0): "仿射+正弦漂移+弹性场\n200 trials / held-out TRE-EPE",
        (1, 1): "CAVE/Harvard/Chikusei\n固定 PSF/SRF 的 Wald protocol",
        (1, 2): "已知 SRF/FWHM 与噪声\n拼接连续性真值",
        (1, 3): "合成材料边界/矿物标签\n避免 patch 泄漏",
        (2, 0): "每对传感器每箱 40–50 地标\n3 名标注者 + bootstrap CI",
        (2, 1): "重复扫描 / HR-NIR-SWIR ROI\n暗20/中亮/亮分层",
        (2, 2): "点光谱 / 白板 / 暗电流\n1400/1900 nm 单独报告",
        (2, 3): "按钻孔/箱独立测试\nXRD/Raman/专家矿物标签",
        (3, 0): "坐标修正、边界拒绝\n列几何、tiepoint、cycle",
        (3, 1): "log vs 线性、exp vs 加法\n门控、零空间、回投影、裁剪",
        (3, 2): "重叠区、平滑、坏波段\n统一谱轴与重复波段",
        (3, 3): "融合前后矿物/岩性/裂隙\n冻结参数与失败案例",
        (4, 0): "TRE median/P95、≤0.5/1 px\nJacobian、cycle、耗时",
        (4, 1): "PSNR/SSIM/SAM/ERGAS/Q2n\n暗部相关、噪声、halo、裁剪率",
        (4, 2): "重叠 RMSE/导数跳变\n吸收中心/带深误差",
        (4, 3): "F1/mIoU/OA、效应量\n置信区间与显著性",
        (5, 0): "合成 P95 <0.5 HSI px\n真实阈值预注册后冻结",
        (5, 1): "V6 显著优于 V5/强基线\n且光谱误差不过度上升",
        (5, 2): "拼接误差下降且无吸收漂移\n低 SNR 波段不强约束",
        (5, 3): "跨钻孔稳定提升\n无单景调参依赖",
    }
    left, bottom, width, height = 0.14, 0.09, 0.83, 0.80
    col_w, row_h = width / len(columns), height / len(rows)
    for col, label in enumerate(columns):
        ax.add_patch(Rectangle((left + col * col_w, bottom + height), col_w, 0.055, transform=ax.transAxes, facecolor=NAVY, edgecolor="white"))
        ax.text(left + (col + 0.5) * col_w, bottom + height + 0.027, label, transform=ax.transAxes, color="white", ha="center", va="center", weight="bold")
    for row, label in enumerate(rows):
        y = bottom + height - (row + 1) * row_h
        ax.add_patch(Rectangle((0.025, y), 0.11, row_h, transform=ax.transAxes, facecolor=BLUE if row < 5 else ORANGE, edgecolor="white"))
        ax.text(0.08, y + row_h / 2, label, transform=ax.transAxes, color="white", ha="center", va="center", weight="bold")
        for col in range(len(columns)):
            face = "#F8FAFC" if (row + col) % 2 == 0 else "#EDF3F8"
            if row == 5:
                face = PALE_ORANGE
            ax.add_patch(Rectangle((left + col * col_w, y), col_w, row_h, transform=ax.transAxes, facecolor=face, edgecolor="white", linewidth=1.2))
            ax.text(left + (col + 0.5) * col_w, y + row_h / 2, cells[(row, col)], transform=ax.transAxes, ha="center", va="center", color=NAVY, fontsize=8.1, linespacing=1.4)
    fig.savefig(out / "09_paper_experiment_matrix.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def roadmap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 5.8))
    ax.axis("off")
    ax.text(0.025, 0.92, "从 V6 工程原型到高水平论文：四阶段路线", fontsize=20, weight="bold", color=NAVY, transform=ax.transAxes)
    phases = [
        ("P0 立即", "代码与证据清障", "坐标/边界修复\n暗部 log-exp 融合\n合成 TRE/EPE", BLUE, PALE_BLUE),
        ("P1 4–6 周", "真值与基线", "人工地标\nGS/PC/NNDiffuse、CNMF、HySure\n公共 HR-HSI Wald", GREEN, PALE_GREEN),
        ("P2 6–12 周", "数据集与冻结测试", "3+ 钻孔、30–50 箱\n重复扫描/点光谱/XRD\n跨孔冻结参数", ORANGE, PALE_ORANGE),
        ("P3 投稿前", "统计与论文闭环", "消融 + 失败案例\n显著性/CI/效应量\n代码数据版本化", NAVY, LIGHT_GRAY),
    ]
    xs = [0.04, 0.285, 0.53, 0.775]
    for x, (tag, title, body, edge, face) in zip(xs, phases, strict=True):
        add_box(ax, (x, 0.30), 0.19, 0.38, f"{tag}\n{title}", body, face=face, edge=edge, title_color=edge)
    for i in range(3):
        arrow(ax, (xs[i] + 0.19, 0.49), (xs[i + 1], 0.49), color=NAVY)
    ax.text(0.04, 0.14, "决策闸门：没有独立真实地标，不声称真实亚像素；没有 HR-HSI/重复扫描，不声称 SWIR 细节无损真值。", transform=ax.transAxes, color=RED, fontsize=11, weight="bold")
    fig.savefig(out / "10_research_roadmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    registration_route(args.output_dir)
    fusion_route(args.output_dir)
    registration_accuracy(summary, args.output_dir)
    fusion_metrics(summary, args.output_dir)
    comparison_figure(args.repo, "3dssz", summary, args.output_dir)
    comparison_figure(args.repo, "zkh3", summary, args.output_dir)
    classic_vs_v6(args.output_dir)
    evidence_ladder(args.output_dir)
    experiment_matrix(args.output_dir)
    roadmap(args.output_dir)
    print("figures=10")


if __name__ == "__main__":
    main()
