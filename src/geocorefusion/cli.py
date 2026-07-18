"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dataset import discover_triplet
from .pipeline import input_summary, run_pipeline, run_registration_review
from .validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geocorefusion", description="RGB-NIR-SWIR drill-core fusion research prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="Inspect a data directory")
    inspect_parser.add_argument("data_dir", type=Path)
    run_parser = sub.add_parser("run", help="Run the complete fusion pipeline")
    run_parser.add_argument("config", type=Path)
    register_parser = sub.add_parser("register", help="Estimate registration only")
    register_parser.add_argument("config", type=Path)
    validate_parser = sub.add_parser("validate", help="Validate a completed run")
    validate_parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(input_summary(discover_triplet(args.data_dir)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_run(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    config = load_config(args.config)
    if args.command == "register":
        result = run_registration_review(config)
        print(json.dumps({
            "output_dir": str(result.output_dir),
            "roi": result.roi,
            "status": result.status,
            "review": str(result.output_dir / "registration_review.json"),
        }, ensure_ascii=False, indent=2))
        return 0
    result = run_pipeline(config)
    print(json.dumps({"output_dir": str(result.output_dir), "roi": result.roi, "status": result.quality_report["summary"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
