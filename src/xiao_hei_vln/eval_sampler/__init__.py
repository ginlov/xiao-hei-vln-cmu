"""Offline eval sample assembler.

Combines a VLA-3D ground-truth JSONL (from dataset_generator) with a
separately-produced predictions JSONL into the evaluator-ready format.

Pipeline:
  1. Load GT JSONL  (VLA-3D format)  → convert answers to VLMOutput
  2. Load pred JSONL ({question, prediction} per line)
  3. Match by question text
  4. Write {question, ground_truth, prediction} JSONL for the evaluator
"""
