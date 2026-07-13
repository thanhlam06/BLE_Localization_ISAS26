"""Create the Baseline ML best-result report from existing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_ml.results import collect_result_rows, write_reports


def main() -> int:
    rows = collect_result_rows(ROOT)
    csv_path, md_path = write_reports(rows, ROOT)
    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    if rows:
        best = rows[0]
        print(
            "best: {score:.6f} from {source}".format(
                score=float(best["mean_macro_f1_candidate"]),
                source=best["source_path"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
