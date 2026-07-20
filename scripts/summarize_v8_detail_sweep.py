"""Summarise the V8 3DSSZ detail-channel Pareto sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RUNS = {
    "V7 frozen": "3dssz_roi_fusion_v7_final_ampfix",
    "coeff 0.020": "3dssz_roi_fusion_v8_sweep_c020",
    "coeff 0.050": "3dssz_roi_fusion_v8_sweep_c050",
    "coeff 0.100": "3dssz_roi_fusion_v8_sweep_c100",
    "coeff 0.050 + gain 1.25": "3dssz_roi_fusion_v8_sweep_c050_g125",
    "coeff 0.050 + additive 0.10": "3dssz_roi_fusion_v8_sweep_c050_a010",
    "ungated gain 1.05": "3dssz_roi_fusion_v8_sweep_ungated_g105",
    "plateau gate + gain 1.05": "3dssz_roi_fusion_v8_sweep_plateau_g105",
    "plateau gate + gain 1.40": "3dssz_roi_fusion_v8_sweep_plateau_g140",
    "plateau gate + no second null + gain 1.05": "3dssz_roi_fusion_v8_sweep_plateau_np0_g105",
    "ungated + no second null + gain 1.00": "3dssz_roi_fusion_v8_sweep_ungated_np0_g100",
    "simplex K6 guidance-off": "3dssz_roi_fusion_v8_sweep_simplex_k6_s000",
    "hybrid K6+R6 guidance-off": "3dssz_roi_fusion_v8_sweep_hybrid_k6r6_s000",
    "hybrid K6+R6 abundance 0.50": "3dssz_roi_fusion_v8_sweep_hybrid_k6r6_s050",
    "bridge rank-1 strength 0.50": "3dssz_roi_fusion_v8_sweep_bridge_r1_s050",
    "bridge rank-2 strength 0.50": "3dssz_roi_fusion_v8_sweep_bridge_r2_s050",
    "bridge rank-2 strength 1.00": "3dssz_roi_fusion_v8_sweep_bridge_r2_s100",
    "local bridge rank-1 strength 0.50": "3dssz_roi_fusion_v8_sweep_local_bridge_r1_s050",
    "local bridge rank-2 strength 0.50": "3dssz_roi_fusion_v8_sweep_local_bridge_r2_s050",
}


def load_row(repo: Path, label: str, run_name: str) -> dict[str, Any]:
    path = repo / "runs" / run_name / "metrics" / "quality_report.json"
    quality = json.loads(path.read_text(encoding="utf-8"))
    spatial = quality["spatial"]["band_detail_by_brightness"]["bands"]
    per_band: list[dict[str, Any]] = []
    for band in ("901.0nm", "1651.0nm", "2201.0nm"):
        report = spatial[band]
        scale = report["multiscale_log_high_frequency"]["sigma_2.4px"]
        reliable = scale["reliable_rgb_detail"]
        dark = scale["dark_reliable_rgb_detail"]
        edge = report["gradient_and_edge"]["reliable_rgb_detail"]
        per_band.append(
            {
                "band": band,
                "rho": reliable["rho"],
                "beta": reliable["beta"],
                "A": reliable["energy_ratio_A"],
                "R_perp": reliable["orthogonal_residual_ratio_R_perp"],
                "dark_beta": dark["beta"],
                "dark_A": dark["energy_ratio_A"],
                "dark_R_perp": dark["orthogonal_residual_ratio_R_perp"],
                "flat_A": scale["rgb_flat"]["energy_ratio_A"],
                "edge_f1": edge["edge_f1_1px"],
            }
        )
    forward = quality["final_hr_product_observation"]
    target = next(item for item in per_band if item["band"] == "2201.0nm")
    mean = lambda key: sum(float(item[key]) for item in per_band) / len(per_band)
    return {
        "label": label,
        "run": run_name,
        "forward_rmse": forward["rmse"],
        "forward_sam_deg": forward["sam_mean_deg"],
        "mean_beta": mean("beta"),
        "mean_A": mean("A"),
        "mean_R_perp": mean("R_perp"),
        "mean_flat_A": mean("flat_A"),
        "mean_edge_f1": mean("edge_f1"),
        "beta_2201": target["beta"],
        "A_2201": target["A"],
        "R_perp_2201": target["R_perp"],
        "dark_beta_2201": target["dark_beta"],
        "dark_A_2201": target["dark_A"],
        "dark_R_perp_2201": target["dark_R_perp"],
        "flat_A_2201": target["flat_A"],
        "edge_f1_2201": target["edge_f1"],
        "all_internal_bounds": all(
            0.8 <= item["beta"] <= 1.2
            and 0.8 <= item["A"] <= 1.25
            and item["R_perp"] <= 0.35
            and 0.8 <= item["dark_beta"] <= 1.2
            and 0.8 <= item["dark_A"] <= 1.25
            and item["dark_R_perp"] <= 0.35
            and item["flat_A"] <= 1.1
            and item["edge_f1"] >= 0.85
            for item in per_band
        ),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "artifacts" / "v7_research" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_row(repo, label, run) for label, run in RUNS.items()]

    csv_path = out_dir / "v8_detail_sweep.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V8 细节通道 Pareto 扫参证据",
        "",
        "所有运行冻结 3DSSZ ROI、配准、光谱谐调、MTF/PSF、子空间秩、forward cycle 与原生 log 评价，只改变细节通道。",
        "",
        "| 方案 | forward RMSE | 平均 β | 平均 A | 平均 R⊥ | 平均 flat A | 平均 Edge F1 | 2201 β | 2201 R⊥ | 2201 暗区 β | 全界内 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    fmt(row["forward_rmse"], 6),
                    fmt(row["mean_beta"]),
                    fmt(row["mean_A"]),
                    fmt(row["mean_R_perp"]),
                    fmt(row["mean_flat_A"]),
                    fmt(row["mean_edge_f1"]),
                    fmt(row["beta_2201"]),
                    fmt(row["R_perp_2201"]),
                    fmt(row["dark_beta_2201"]),
                    "✓" if row["all_internal_bounds"] else "—",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 决策",
        "",
        "- 把 local coefficient residual 从 0.001 提高到 0.100，只带来很小的 β 增益，并增加 R⊥；现有 PCA 系数通道不是缺少一个更大的强度参数。",
        "- 提高公共 log-gain 可把 2201 nm β 推到 0.8 以上，但 R⊥、暗区能量和部分波段过冲同步恶化；这验证了一个公共增益无法适配材料和波段差异。",
        "- 保幅 plateau gate 比连续置信度衰减更能保留暗区相干纹理，同时比完全无门控更能抑制 flat 区；但它不能解决 band-specific 幅值和 RGB-exclusive 纹理问题。",
        "- 取消第二次近零空间投影并未释放预期幅值：plateau/no-second-null 的 2201 nm β=0.562，forward RMSE=0.004231，仍弱于 V7 的 0.565/0.003923；因此瓶颈不是一个重复投影，而是跨模态关系本身。",
        "- 0/19 方案通过旧的 RGB-equivalent 内部筛查。该筛查现仅保留为显示诊断，不能再把每个波段 β≈1 当科学硬目标；即便如此，结果仍证明继续堆叠公共 gain/GSA/PCA 强度没有形成联合 Pareto 点。共享 simplex abundance residual 和低秩 bridge 也只能作为下一阶段受约束原型，不能提前写成已达标。",
        "- 纯 K=6 simplex 的 forward RMSE 为 0.0130；增加 6 个禁止接收 RGB 的 observation-residual PCA 分量后降到约 0.0060，但仍未达到 V7。闭式 abundance 注入在 3DSSZ 的 RGB→丰度 held-out R² 仅约 0.36，并在 2201 nm 出现对比反转，因此当前只保留为研究原型。",
        "- rank-1/2 blocked-CV bridge 的 guidance-off/弱注入 forward RMSE 可到约 0.00385，但 LR 映射的方差加权 R² 仅约 0.096；提高强度会再次造成暗区与 flat 区过冲。它证明低秩限制更安全，但尚不是最终细节解。",
        "- 当前证据支持的下一步不是继续盲扫强度，而是补齐 band-pass blocked-CV 可辨识性、材料条件 log-shading、注册 covariance 与独立 Wald/HR 真值；关系不可辨识的 2201 nm 区域必须允许关闭 RGB 注入。",
        "",
        "这些 RGB 引导指标仍是同数据结构诊断，不是独立 HR-SWIR 真值。",
    ]
    md_path = out_dir / "v8_detail_sweep_interpretation.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
