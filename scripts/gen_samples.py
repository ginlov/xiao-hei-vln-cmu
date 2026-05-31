#!/usr/bin/env python3
"""Assemble evaluator-ready JSONL from GT + predictions (offline).

Usage:
    python scripts/gen_samples.py \
        --gt  vla3d_ref.jsonl   \
        --pred predictions.jsonl \
        --out  eval_samples.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xiao_hei_vln.eval_sampler.__main__ import main

if __name__ == "__main__":
    main()
