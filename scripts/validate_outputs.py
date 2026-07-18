from __future__ import annotations

import argparse
import json

from geocorefusion.validation import validate_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(validate_run(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

