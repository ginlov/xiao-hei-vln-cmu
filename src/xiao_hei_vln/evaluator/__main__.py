"""CLI entry point: python -m xiao_hei_vln.evaluator <samples.jsonl> [--json]

Each line of the JSONL file must be a JSON object with:
  question     : str
  ground_truth : VLMOutput dict  (e.g. {"kind": "numerical", "value": 3})
  prediction   : VLMOutput dict

Example line:
  {"question": "How many chairs?", "ground_truth": {"kind": "numerical", "value": 3},
   "prediction": {"kind": "numerical", "value": 4}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from xiao_hei_vln.evaluator.runner import Evaluator
from xiao_hei_vln.evaluator.types import EvalSample
from xiao_hei_vln.messages.outputs import parse_vlm_output


def load_samples(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(
                    EvalSample(
                        question=obj["question"],
                        ground_truth=parse_vlm_output(obj["ground_truth"]),
                        prediction=parse_vlm_output(obj["prediction"]),
                    )
                )
            except Exception as exc:
                print(f"[warn] line {lineno}: {exc}", file=sys.stderr)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM predictions against ground truth (Task 1 & 2)."
    )
    parser.add_argument("samples", type=Path, help="Path to a JSONL file of eval samples.")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON instead of human-readable."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write JSON results to this file (implies --json)."
    )
    args = parser.parse_args()

    samples = load_samples(args.samples)
    if not samples:
        print("No valid samples found.", file=sys.stderr)
        sys.exit(1)

    report = Evaluator().evaluate(samples)

    if args.out:
        args.out.write_text(report.to_json())
        print(f"Results written to {args.out}")
        report.print_summary()
    elif args.json:
        print(report.to_json())
    else:
        report.print_summary()


if __name__ == "__main__":
    main()
