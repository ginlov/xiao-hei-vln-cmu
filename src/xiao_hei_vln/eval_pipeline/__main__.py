"""End-to-end evaluation pipeline CLI.

Combines sample assembly (GT + predictions → EvalSample) and evaluation in a
single command, without writing an intermediate JSONL file to disk.

Usage:
    python -m xiao_hei_vln.eval_pipeline \\
        --gt  vla3d_ref.jsonl   \\
        --pred predictions.jsonl

    # save JSON results to a file
    python -m xiao_hei_vln.eval_pipeline \\
        --gt  vla3d_ref.jsonl   \\
        --pred predictions.jsonl \\
        --out results.json

    # also keep the intermediate samples JSONL for debugging
    python -m xiao_hei_vln.eval_pipeline \\
        --gt  vla3d_ref.jsonl   \\
        --pred predictions.jsonl \\
        --samples-out eval_samples.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xiao_hei_vln.eval_sampler.__main__ import assemble_samples, write_samples_jsonl
from xiao_hei_vln.evaluator.runner import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble eval samples and evaluate in one step (no intermediate file)."
    )
    parser.add_argument("--gt", type=Path, required=True, help="VLA-3D GT JSONL file.")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Max GT entries to load.")
    parser.add_argument(
        "--json", action="store_true", help="Print results as JSON instead of human-readable."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write JSON results to this file (implies --json)."
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=None,
        help="Also write assembled samples to this JSONL path (useful for debugging).",
    )
    args = parser.parse_args()

    samples = assemble_samples(args.gt, args.pred, args.limit)
    if not samples:
        print("No valid samples found.", file=sys.stderr)
        sys.exit(1)

    if args.samples_out:
        write_samples_jsonl(samples, args.samples_out)
        print(f"Samples written to {args.samples_out}", file=sys.stderr)

    print("\n--- Evaluation ---", file=sys.stderr)
    report = Evaluator().evaluate(samples)

    if args.out:
        args.out.write_text(report.to_json())
        print(f"Results written to {args.out}", file=sys.stderr)
        report.print_summary()
    elif args.json:
        print(report.to_json())
    else:
        report.print_summary()


if __name__ == "__main__":
    main()
