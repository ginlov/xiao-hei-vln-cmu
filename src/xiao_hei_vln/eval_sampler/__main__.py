"""Offline eval sample assembler CLI.

Combines a VLA-3D ground-truth JSONL with a separately-produced predictions
JSONL into the format expected by the evaluator. No model calls are made.

Usage:
    python -m xiao_hei_vln.eval_sampler \\
        --gt  vla3d_ref.jsonl   \\
        --pred predictions.jsonl \\
        --out  eval_samples.jsonl

Input formats:
  --gt   VLA-3D JSONL (from dataset_generator): each line has
         {type, question, answer, object_list, ...}

  --pred Predictions JSONL: each line has
         {"question": "...", "prediction": {<VLMOutput dict>}}

Output:
  One line per matched pair:
  {"question": "...", "ground_truth": {<VLMOutput>}, "prediction": {<VLMOutput>}}

  → feed directly to: python -m xiao_hei_vln.evaluator eval_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from xiao_hei_vln.eval_sampler.gt_converter import gt_from_entry
from xiao_hei_vln.evaluator.types import EvalSample
from xiao_hei_vln.messages.outputs import VLMOutput, parse_vlm_output


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno}: {exc}", file=sys.stderr)
    return entries


def assemble_samples(
    gt_path: Path, pred_path: Path, limit: int | None
) -> list[EvalSample]:
    """Build EvalSample objects in memory from GT and predictions files.

    Called by both the gen-samples CLI (which then writes to disk) and the
    eval-pipeline CLI (which passes results directly to the evaluator).
    """
    gt_entries = _load_jsonl(gt_path)
    if limit is not None:
        gt_entries = gt_entries[:limit]

    # Fix #6: correct type annotation (was dict[str, object])
    gt_lookup: dict[str, VLMOutput] = {}
    gt_skipped = 0
    for entry in gt_entries:
        question = entry.get("question", "").strip()
        if not question:
            gt_skipped += 1
            continue
        gt = gt_from_entry(entry)
        if gt is None:
            print(
                f"  [skip GT] {question!r:.60}: could not convert GT (type={entry.get('type')!r})",
                file=sys.stderr,
            )
            gt_skipped += 1
            continue
        # Fix #4: warn on duplicate question keys instead of silently overwriting
        if question in gt_lookup:
            print(
                f"  [dup GT] duplicate question key — keeping first: {question!r:.60}",
                file=sys.stderr,
            )
            gt_skipped += 1
            continue
        gt_lookup[question] = gt

    print(f"GT loaded: {len(gt_lookup)} entries  (skipped {gt_skipped})", file=sys.stderr)
    if limit is not None:
        print(f"  (note: --limit {limit} is active; predictions outside this window will show as unmatched)", file=sys.stderr)

    pred_entries = _load_jsonl(pred_path)
    print(f"Predictions loaded: {len(pred_entries)} entries", file=sys.stderr)

    samples: list[EvalSample] = []
    unmatched = parse_errors = 0
    for i, pred_entry in enumerate(pred_entries):
        question = pred_entry.get("question", "").strip()

        gt = gt_lookup.get(question)
        if gt is None:
            print(f"  [no GT] entry {i}: {question!r:.60}", file=sys.stderr)
            unmatched += 1
            continue

        # Fix #5: handle missing "prediction" key explicitly before parse
        raw_pred = pred_entry.get("prediction")
        if raw_pred is None:
            print(f"  [missing key] entry {i}: no 'prediction' field", file=sys.stderr)
            parse_errors += 1
            continue
        try:
            pred = parse_vlm_output(raw_pred)
        except Exception as exc:
            print(f"  [parse error] entry {i}: {exc}", file=sys.stderr)
            parse_errors += 1
            continue

        samples.append(EvalSample(question=question, ground_truth=gt, prediction=pred))

    print(f"\nAssembled {len(samples)} samples  (unmatched={unmatched}  parse_errors={parse_errors})", file=sys.stderr)
    return samples


def write_samples_jsonl(samples: list[EvalSample], out_path: Path) -> None:
    """Serialize EvalSamples to a JSONL file (shared by gen-samples and eval-pipeline CLIs)."""
    with out_path.open("w") as out_f:
        for sample in samples:
            record = {
                "question": sample.question,
                "ground_truth": sample.ground_truth.model_dump(),
                "prediction": sample.prediction.model_dump(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(gt_path: Path, pred_path: Path, out_path: Path, limit: int | None) -> None:
    samples = assemble_samples(gt_path, pred_path, limit)
    write_samples_jsonl(samples, out_path)
    print(f"Output: {out_path}  ({len(samples)} samples)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble evaluator-ready JSONL from GT + predictions (offline)."
    )
    parser.add_argument("--gt", type=Path, required=True, help="VLA-3D GT JSONL file.")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--out", type=Path, required=True, help="Output eval JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Max GT entries to load.")
    args = parser.parse_args()

    run(gt_path=args.gt, pred_path=args.pred, out_path=args.out, limit=args.limit)


if __name__ == "__main__":
    main()
