"""Create a compact CSV/Markdown table from saved baseline metric artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_run(value: str) -> tuple[str, Path, Path | None]:
    """Parse NAME=METRICS_PATH[,CONFIG_PATH]."""
    if "=" not in value:
        raise ValueError("Each --run must use NAME=METRICS_PATH[,CONFIG_PATH]")
    name, paths = value.split("=", 1)
    parts = paths.split(",", 1)
    return name, Path(parts[0]), Path(parts[1]) if len(parts) == 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=METRICS_PATH[,CONFIG_PATH]; repeat for each baseline.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "reports")
    args = parser.parse_args()
    rows = []
    for value in args.run:
        name, metrics_path, config_path = parse_run(value)
        metrics = read_json(metrics_path)
        config = read_json(config_path) if config_path and config_path.exists() else {}
        decision_errors = metrics.get("decision_error_rates") or {}
        retrieval_hit = next((value for key, value in metrics.items() if key.startswith("evidence_hit_at_")), None)
        rows.append(
            {
                "system": name,
                "evidence_mode": config.get("evidence_mode", "not_recorded"),
                "claim_count": metrics.get("claim_count", ""),
                "evidence_hit_at_3": retrieval_hit if retrieval_hit is not None else "",
                "mrr": metrics.get("mrr", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "accuracy": metrics.get("accuracy", ""),
                "false_reassurance_rate": decision_errors.get("false_reassurance_rate", ""),
                "false_alarm_rate": decision_errors.get("false_alarm_rate", ""),
                "metrics_path": str(metrics_path),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["# Baseline summary", "", "| " + " | ".join(fieldnames[:-1]) + " |", "| " + " | ".join(["---"] * (len(fieldnames) - 1)) + " |"]
    for row in rows:
        markdown.append("| " + " | ".join(str(row[field]) for field in fieldnames[:-1]) + " |")
    (args.output_dir / "baseline_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
