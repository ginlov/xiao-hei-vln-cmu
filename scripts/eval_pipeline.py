#!/usr/bin/env python3
"""End-to-end evaluation pipeline: assemble samples and evaluate in one step.

Usage:
    python scripts/eval_pipeline.py \
        --gt  vla3d_ref.jsonl   \
        --pred predictions.jsonl

    # save JSON results
    python scripts/eval_pipeline.py \
        --gt  vla3d_ref.jsonl   \
        --pred predictions.jsonl \
        --out  results.json

    # also keep the intermediate samples JSONL for debugging
    python scripts/eval_pipeline.py \
        --gt  vla3d_ref.jsonl   \
        --pred predictions.jsonl \
        --samples-out eval_samples.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xiao_hei_vln.eval_pipeline.__main__ import main

if __name__ == "__main__":
    main()
