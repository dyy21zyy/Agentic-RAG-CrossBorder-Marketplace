"""Evaluation utilities for metrics, datasets, runners, ablations, and reports."""
from crossborder_agentic_rag.evaluation.datasets import EvalExample, load_eval_jsonl, write_eval_jsonl
from crossborder_agentic_rag.evaluation.evaluator import EvalResult, EvalSummary, evaluate_agent
from crossborder_agentic_rag.evaluation.ablations import AblationConfig, AblationResult, default_ablation_configs, run_ablations
