"""Re-run matched V5/V6/V7 ROIs with the amplitude-safe V7 evaluator.

The script deliberately writes new run directories.  It never overwrites the
historical V5/V6/V7 evidence used to diagnose the percentile-normalisation
bug in beta/A/R_perp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


RUN_SPECS = {
    "3dssz_v5": (
        "configs/3dssz_roi_fusion_v5_matched.yaml",
        "3dssz_roi_fusion_v5_matched_v7eval_ampfix",
    ),
    "3dssz_v6": (
        "configs/3dssz_roi_fusion_v6.yaml",
        "3dssz_roi_fusion_v6_v7eval_ampfix",
    ),
    "3dssz_v7": (
        "configs/3dssz_roi_fusion_v7.yaml",
        "3dssz_roi_fusion_v7_final_ampfix",
    ),
    "zkh3_v5": (
        "configs/zkh3_roi_fusion_v5_matched.yaml",
        "zkh3_roi_fusion_v5_matched_v7eval_ampfix",
    ),
    "zkh3_v6": (
        "configs/zkh3_roi_fusion_v6.yaml",
        "zkh3_roi_fusion_v6_v7eval_ampfix",
    ),
    "zkh3_v7": (
        "configs/zkh3_roi_fusion_v7.yaml",
        "zkh3_roi_fusion_v7_final_ampfix",
    ),
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run matched ROIs with native-log amplitude metrics."
    )
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(RUN_SPECS),
        default=list(RUN_SPECS),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly refresh the script-owned ampfix run directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from geocorefusion.config import load_config
    from geocorefusion.pipeline import run_pipeline

    results: list[dict[str, object]] = []
    for variant in args.variants:
        config_rel, output_name = RUN_SPECS[variant]
        config = load_config(repo / config_rel)
        config.output_dir = repo / "runs" / output_name
        # The factorised product is now exactly reconstructable without the
        # approximately 2.3 GiB full cube; keep the benchmark compact.
        config.output.write_envi = False
        config.output.write_coefficients = True
        config.output.overwrite_files = bool(args.overwrite_existing)
        started = time.perf_counter()
        result = run_pipeline(config)
        elapsed = time.perf_counter() - started
        forward = result.quality_report["final_hr_product_observation"]
        item = {
            "variant": variant,
            "output_dir": str(result.output_dir),
            "elapsed_seconds": elapsed,
            "summary_status": result.quality_report["summary"]["status"],
            "forward_rmse": forward["rmse"],
            "forward_sam_mean_deg": forward["sam_mean_deg"],
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
