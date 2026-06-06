#!/usr/bin/env python3
"""
PRISM Stage 2 — Recommender Batch Runner with GPU Scheduling & Live Dashboard

Auto-discovers Stage 1 experiment outputs, runs Stage 2 recommender training
for each, with the same GPU-parallel scheduler and rich dashboard as Stage 1.

Usage:
  python scripts/prism/batch/stage2.py [DATASET] [--gpus 0,1,2,3]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── project paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
REC_TRAIN_MODULE = "src.recommender.prism.train"

DATASET_MAP = {
    "beauty": "Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
    "sports": "Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports",
    "toys": "Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys",
    "cds": "Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
}

# If empty, all valid stage1 subdirectories are run.
# Override via CLI: --experiments exp1,exp2,exp3
TARGET_STAGE1_EXPERIMENTS = []

# TCAF ablation (disabled — use clean DenseRouter MoE for now)
TCAF_ABLATION = []

# Sparse MoE ablation: test (num_experts, top_k) combinations
# against the dense MoE baseline.  Each entry: (name, stage1_variant, extra_rec_args).
# stage1_variant=None means "use the first auto-discovered stage1 experiment".
# Controlled by --ablation sparse_moe (or set ABLATION env var).
SPARSE_MOE_ABLATION = [
    ("sparse_moe_e3k1",  None, ["--moe_num_experts", "3", "--moe_top_k", "1"]),
    ("sparse_moe_e3k2",  None, ["--moe_num_experts", "3", "--moe_top_k", "2"]),
    ("sparse_moe_e5k2",  None, ["--moe_num_experts", "5", "--moe_top_k", "2"]),
    ("sparse_moe_e5k3",  None, ["--moe_num_experts", "5", "--moe_top_k", "3"]),
]

# Active ablation mode (set via --ablation or ABLATION env var).
# "" = normal auto-discovery mode (stage1 variants × fixed stage2 config).
# "sparse_moe" = single stage1 × multiple stage2 sparse MoE configs.
ABLATION_MODE = os.environ.get("ABLATION", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecExperiment:
    name: str
    stage1_dir: Path
    output_dir: Path
    semantic_map: Path
    purified_content: Path
    purified_collab: Path
    purified_dim: int = 128
    status: str = "queued"
    gpu: int = -1
    proc: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # metrics from stage1
    stage1_val_loss: float = 0.0
    stage1_pre_unique_rate: float = 0.0
    stage1_collision_item_rate: float = 0.0
    stage1_knn_l1: float = 0.0
    stage1_knn_l12: float = 0.0
    stage1_prefix_l1: float = 0.0
    stage1_prefix_l2: float = 0.0
    stage1_prefix_l3: float = 0.0
    stage1_code_usage_avg: float = 0.0
    stage1_seq_l1: float = 0.0
    stage1_seq_l2: float = 0.0
    stage1_seq_l3: float = 0.0
    stage1_seq_depth: float = 0.0
    # metrics from stage2 log (best test-set)
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    test_recall10: float = 0.0
    test_recall20: float = 0.0
    test_ndcg10: float = 0.0
    test_ndcg20: float = 0.0
    best_metric: float = 0.0
    best_metric_name: str = ""
    best_epoch: int = 0
    extra_rec_args: List[str] = field(default_factory=list)
    _log_f: object = None


# ═══════════════════════════════════════════════════════════════════════════════
# Log Parser
# ═══════════════════════════════════════════════════════════════════════════════

_EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)")
_TRAIN_RE = re.compile(r"Training - Total: ([\d.]+), Main: ([\d.]+)")
_PRED_RE = re.compile(r"Pred: ([\d.]+)")
_RECALL10_RE = re.compile(r"Recall@10: ([\d.]+)")
_RECALL20_RE = re.compile(r"Recall@20: ([\d.]+)")
_NDCG10_RE = re.compile(r"NDCG@10: ([\d.]+)")
_NDCG20_RE = re.compile(r"NDCG@20: ([\d.]+)")
_BEST_EPOCH_RE = re.compile(r"Best epoch: (\d+), Best (\S+): ([\d.]+)")

# Test block format:
#   Test: Recall Metrics:\n  Recall@10: 0.0396\n  ...\nNDCG Metrics:\n  NDCG@10: ...\n  ...\nSemantic ID...\n...
#   Terminated by blank line or section separator
_TEST_BLOCK_RE = re.compile(r"Test: .*?Recall@10: ([\d.]+).*?Recall@20: ([\d.]+).*?NDCG@10: ([\d.]+).*?NDCG@20: ([\d.]+)", re.DOTALL)


def parse_rec_metrics(log_path: Path) -> Dict:
    """Scan the recommender training log and extract best test-set metrics."""
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text()
    except Exception:
        return {}

    metrics = {}

    # current epoch progress
    epochs = _EPOCH_RE.findall(text)
    if epochs:
        metrics["epoch"] = int(epochs[-1][0])
        metrics["total_epochs"] = int(epochs[-1][1])

    # latest training loss in the last epoch block
    epoch_starts = list(_EPOCH_RE.finditer(text))
    if epoch_starts:
        block = text[epoch_starts[-1].start():]
        m = _TRAIN_RE.search(block)
        if m:
            metrics["train_loss"] = float(m.group(1))

    # parse the LAST Test: block — test set is evaluated only on new-best epochs
    # format: "Test: Recall Metrics:\n  Recall@10: X\n  ...\n NDCG Metrics:\n  ..."
    # terminated by blank line or INFO separator
    test_idx = text.rfind("Test: Recall Metrics:")
    if test_idx >= 0:
        tail = text[test_idx:]
        # find end of block (blank line or new section)
        end_match = re.search(r"\n\s*\n", tail)
        block = tail[:end_match.start()] if end_match else tail

        for pat, key in [(_RECALL10_RE, "test_recall10"),
                          (_RECALL20_RE, "test_recall20"),
                          (_NDCG10_RE, "test_ndcg10"),
                          (_NDCG20_RE, "test_ndcg20")]:
            m = pat.search(block)
            if m:
                metrics[key] = float(m.group(1))

    # final best epoch summary
    m = _BEST_EPOCH_RE.search(text)
    if m:
        metrics["best_epoch"] = int(m.group(1))
        metrics["best_metric_name"] = m.group(2)
        metrics["best_metric"] = float(m.group(3))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpus(requested: Optional[List[int]] = None) -> List[int]:
    try:
        import torch
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if requested is not None:
                return [g for g in requested if g in all_gpus]
            return all_gpus
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = [int(l.strip()) for l in result.stdout.strip().split("\n") if l.strip()]
            if requested is not None:
                return [g for g in requested if g in gpus]
            return gpus
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

console = Console()

STATUS_ICONS = {"queued": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "skipped": "⏭️"}
STATUS_COLORS = {"queued": "dim", "running": "yellow", "done": "green", "failed": "red", "skipped": "dim"}


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_dashboard(exps: List[RecExperiment], gpus: List[int],
                    running_per_gpu: Dict[int, Optional[RecExperiment]],
                    start_time: float) -> Layout:
    n_queued = sum(1 for e in exps if e.status == "queued")
    n_running = sum(1 for e in exps if e.status == "running")
    n_done = sum(1 for e in exps if e.status == "done")
    n_failed = sum(1 for e in exps if e.status == "failed")
    n_skipped = sum(1 for e in exps if e.status == "skipped")
    elapsed = format_duration(time.time() - start_time)

    # compact header (1 line + GPU info)
    gpu_parts = []
    for gpu_id in gpus:
        exp = running_per_gpu.get(gpu_id)
        if exp:
            gpu_parts.append(f"GPU{gpu_id}:[yellow]{exp.name}[/yellow]")
        else:
            gpu_parts.append(f"GPU{gpu_id}:[dim]idle[/dim]")
    gpu_line = "  ".join(gpu_parts)

    header_text = Text()
    header_text.append(f"PRISM Stage 2 — Rec  |  {elapsed}  |  ", style="bold magenta")
    header_text.append(f"{n_running} run", style="yellow")
    header_text.append(f"  {n_queued} queued")
    header_text.append(f"  {n_done} done", style="green")
    if n_failed:
        header_text.append(f"  {n_failed} failed", style="red")
    header_text.append(f"\n{gpu_line}")

    layout = Layout()
    layout.split(
        Layout(name="header", size=3 + max(0, len(gpus) - 4)),
        Layout(name="body"),
        Layout(name="footer", size=1),
    )
    layout["header"].update(Panel(header_text, box=box.ROUNDED))

    # experiment table
    exp_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", padding=(0, 1))
    exp_table.add_column("#", width=3, style="dim")
    exp_table.add_column("Experiment", width=20)
    exp_table.add_column("S1 Val", width=8, justify="right")
    exp_table.add_column("S1 PreUniq", width=9, justify="right")
    exp_table.add_column("S1 Coll", width=7, justify="right")
    exp_table.add_column("kNN", width=9, justify="right")
    exp_table.add_column("Seq", width=9, justify="right")
    exp_table.add_column("Status", width=8)
    exp_table.add_column("Epoch", width=7, justify="right")
    exp_table.add_column("Tr Loss", width=8, justify="right")
    exp_table.add_column("Best", width=6, justify="right")
    exp_table.add_column("R@10", width=7, justify="right")
    exp_table.add_column("R@20", width=7, justify="right")
    exp_table.add_column("N@10", width=7, justify="right")
    exp_table.add_column("N@20", width=7, justify="right")
    exp_table.add_column("GPU", width=4, justify="center")
    exp_table.add_column("Time", width=8)

    for i, exp in enumerate(exps):
        icon = STATUS_ICONS.get(exp.status, "?")
        color = STATUS_COLORS.get(exp.status, "")

        if exp.status == "running" and exp.started_at:
            dur = format_duration(time.time() - exp.started_at)
        elif exp.finished_at and exp.started_at:
            dur = format_duration(exp.finished_at - exp.started_at)
        else:
            dur = "--"

        gpu_label = str(exp.gpu) if exp.gpu >= 0 else "--"

        def _f(v, d=4):
            return f"{v:.{d}f}" if v else "--"

        exp_table.add_row(
            str(i + 1), exp.name,
            _f(exp.stage1_val_loss) if exp.stage1_val_loss else "--",
            f"{exp.stage1_pre_unique_rate:.1%}" if exp.stage1_pre_unique_rate else "--",
            f"{exp.stage1_collision_item_rate:.1%}" if exp.stage1_collision_item_rate else "--",
            f"{exp.stage1_knn_l1:.2f}/{exp.stage1_knn_l12:.2f}" if exp.stage1_knn_l1 else "--",
            f"{exp.stage1_seq_l1:.2f}/{exp.stage1_seq_l2:.2f}" if exp.stage1_seq_l1 else "--",
            f"[{color}]{icon} {exp.status}[/{color}]",
            f"{exp.epoch}/{exp.total_epochs}" if exp.epoch else "--",
            _f(exp.train_loss),
            f"{exp.best_metric:.4f}" if exp.best_metric else "--",
            _f(exp.test_recall10),
            _f(exp.test_recall20),
            _f(exp.test_ndcg10),
            _f(exp.test_ndcg20),
            gpu_label, dur,
        )

    layout["body"].update(exp_table)

    footer_text = Text()
    footer_text.append("Logs: <exp>/training.log", style="dim")
    footer_text.append("  |  ")
    footer_text.append("Ctrl+C", style="bold")
    footer_text.append(" to stop", style="dim")
    layout["footer"].update(Panel(footer_text, box=box.ROUNDED, padding=(0, 2)))

    return layout


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Stage 2 arguments (mirrors the bash script defaults)
# ═══════════════════════════════════════════════════════════════════════════════

BASE_REC_ARGS = [
    "--device", "cuda:0",
    "--num_workers", "4",
    "--model_type", "t5-tiny-2",
    "--log_clean",
    "--use_multimodal_fusion",
    "--fusion_gate_type", "dense",
    "--use_purified_predictor",
    "--purified_predictor_weight", "0.1",
    "--use_item_layer_emb", "--use_temporal_decay",
    "--use_trie_constraints",
    "--use_adaptive_temperature",
    "--tau_alpha", "0.5", "--tau_min", "0.7", "--tau_max", "0.8", "--tau_start_layer", "1",
    "--lr_scheduler", "warmup_cosine",
    "--eval_every_n_epochs", "3",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class RecBatchRunner:
    def __init__(self, dataset: str, gpus: Optional[List[int]] = None,
                 output_base: Optional[Path] = None,
                 fast_dev_config: Optional[Path] = None,
                 stage1_base: Optional[Path] = None,
                 experiments: Optional[List[str]] = None):
        self.dataset = dataset
        self.gpus = detect_gpus(gpus)
        if not self.gpus:
            print("ERROR: No GPUs detected!")
            sys.exit(1)

        if dataset not in DATASET_MAP:
            print(f"ERROR: Unknown dataset '{dataset}'. Choose from {list(DATASET_MAP.keys())}")
            sys.exit(1)

        self.project_root = PROJECT_ROOT
        self.stage1_base = stage1_base or (
            PROJECT_ROOT / "scripts/output/prism_tokenizer" / dataset / "hparam_stage1")
        self.output_base = output_base or (
            PROJECT_ROOT / "scripts/output/recommender/prism" / dataset / "hparam_stage1Rec"
        )
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.experiments: List[RecExperiment] = []
        self.gpu_queue: List[int] = list(self.gpus)
        self.running: Dict[int, RecExperiment] = {}
        self.start_time = time.time()
        self.target_experiments = self._resolve_target_experiments(
            experiments if experiments else TARGET_STAGE1_EXPERIMENTS)
        self.fast_dev_config = fast_dev_config

    @staticmethod
    def _normalize_experiment_name(name: str) -> str:
        return EXPERIMENT_NAME_ALIASES.get(name.strip().lower(), name.strip())

    def _resolve_target_experiments(self, names: List[str]) -> List[str]:
        """If names is empty, auto-discover all valid dirs in stage1_base.
        In ablation mode, skip auto-discovery entirely."""
        if ABLATION_MODE:
            return []
        if not names:
            if not self.stage1_base.exists():
                print(f"ERROR: Stage1 output dir not found: {self.stage1_base}")
                sys.exit(1)
            names = sorted([
                d.name for d in self.stage1_base.iterdir()
                if d.is_dir() and (d / "semantic_id_mappings.json").exists()
            ])
            print(f"Auto-discovered {len(names)} stage1 experiments: {names}")
        return names

    def discover_experiments(self):
        """Scan stage1 output dirs and enqueue valid experiments."""
        if not self.stage1_base.exists():
            print(f"ERROR: Stage1 output dir not found: {self.stage1_base}")
            sys.exit(1)

        stage1_dirs = {
            d.name: d for d in self.stage1_base.iterdir()
            if d.is_dir()
        }

        for exp_name in self.target_experiments:
            d = stage1_dirs.get(exp_name)
            if d is None:
                print(f"  [skip] {exp_name}: missing stage1 output dir")
                continue

            semantic_map = d / "semantic_id_mappings.json"
            purified_content = d / "item_purified_content.npy"
            purified_collab = d / "item_purified_collab.npy"

            if not semantic_map.exists():
                print(f"  [skip] {exp_name}: missing semantic_id_mappings.json")
                continue

            output_dir = self.output_base / exp_name
            exp = RecExperiment(
                name=exp_name,
                stage1_dir=d,
                output_dir=output_dir,
                semantic_map=semantic_map,
                purified_content=purified_content,
                purified_collab=purified_collab,
                purified_dim=128,
                log_path=output_dir / "training.log",
            )

            # load stage1 config for purified_dim
            ckpt_path = d / "best_model.pt"
            if ckpt_path.exists():
                try:
                    import torch
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    cfg = ckpt.get("config", {})
                    exp.purified_dim = cfg.get("ide_dim", 128)
                except Exception:
                    pass

            # load stage1 history for context
            hist_path = d / "training_history.json"
            if hist_path.exists():
                try:
                    hist = json.loads(hist_path.read_text())
                    if "val_total_loss" in hist:
                        exp.stage1_val_loss = hist["val_total_loss"][-1]
                except Exception:
                    pass
            sid_quality_path = d / "sid_quality_pre_sinkhorn.json"
            reassign_path = d / "reassignment_stats.json"
            try:
                if sid_quality_path.exists():
                    sidq = json.loads(sid_quality_path.read_text())
                    exp.stage1_pre_unique_rate = sidq.get("unique_rate", 0.0)
                    exp.stage1_collision_item_rate = sidq.get("items_in_collision_rate", 0.0)
                    exp.stage1_knn_l1 = sidq.get("knn_l1", 0.0)
                    exp.stage1_knn_l12 = sidq.get("knn_l12", 0.0)
                    prefix = sidq.get("prefix_unique_rate", {})
                    exp.stage1_prefix_l1 = prefix.get("L1", 0.0)
                    exp.stage1_prefix_l2 = prefix.get("L2", 0.0)
                    exp.stage1_prefix_l3 = prefix.get("L3", 0.0)
                    usage = sidq.get("code_usage_rate", {})
                    if usage:
                        exp.stage1_code_usage_avg = sum(usage.values()) / len(usage)
                    seq = sidq.get("sequence_prefix_hit_rate", {}).get("valid", {})
                    if not seq:
                        seq = sidq.get("sequence_prefix_hit_rate", {}).get("train", {})
                    exp.stage1_seq_l1 = seq.get("L1_prefix_hit", 0.0)
                    exp.stage1_seq_l2 = seq.get("L2_prefix_hit", 0.0)
                    exp.stage1_seq_l3 = seq.get("L3_prefix_hit", 0.0)
                    exp.stage1_seq_depth = seq.get("avg_max_prefix_depth_norm", 0.0)
                elif reassign_path.exists():
                    before = json.loads(reassign_path.read_text()).get("before", {})
                    exp.stage1_pre_unique_rate = before.get("uniqueness_rate", 0.0)
                    total = before.get("total_items", 0) or 0
                    affected = before.get("items_in_collisions", 0) or 0
                    exp.stage1_collision_item_rate = affected / total if total else 0.0
            except Exception:
                pass

            # skip if already completed — load metrics from log
            if (output_dir / "best_model.pt").exists():
                exp.status = "skipped"
                try:
                    import torch
                    ckpt = torch.load(output_dir / "best_model.pt",
                                      map_location="cpu", weights_only=False)
                    exp.best_metric = ckpt.get("best_metric", 0.0)
                except Exception:
                    pass
                # also load test metrics from log for dashboard display
                log_m = parse_rec_metrics(self._resolve_log_path(exp))
                for k, v in log_m.items():
                    setattr(exp, k, v)

            self.experiments.append(exp)

        # TCAF ablation experiments
        for tcaf_name, stage1_variant, extra_args in TCAF_ABLATION:
            d = stage1_dirs.get(stage1_variant)
            if d is None:
                print(f"  [skip] tcaf/{tcaf_name}: stage1 variant '{stage1_variant}' not found")
                continue

            semantic_map = d / "semantic_id_mappings.json"
            purified_content = d / "item_purified_content.npy"
            purified_collab = d / "item_purified_collab.npy"

            if not semantic_map.exists():
                print(f"  [skip] tcaf/{tcaf_name}: missing semantic_id_mappings.json")
                continue

            output_dir = self.output_base / tcaf_name
            exp = RecExperiment(
                name=tcaf_name,
                stage1_dir=d,
                output_dir=output_dir,
                semantic_map=semantic_map,
                purified_content=purified_content,
                purified_collab=purified_collab,
                purified_dim=128,
                log_path=output_dir / "training.log",
                extra_rec_args=extra_args,
            )

            if (output_dir / "best_model.pt").exists():
                exp.status = "skipped"
                log_m = parse_rec_metrics(self._resolve_log_path(exp))
                for k, v in log_m.items():
                    setattr(exp, k, v)

            self.experiments.append(exp)

        # Ablation mode: single stage1 × multiple stage2 configs.
        # Only active when --ablation sparse_moe is set (or ABLATION env var).
        if ABLATION_MODE == "sparse_moe":
            for moe_name, stage1_variant, extra_args in SPARSE_MOE_ABLATION:
                if stage1_variant is None:
                    if not stage1_dirs:
                        print(f"  [skip] sparse_moe/{moe_name}: no stage1 experiments found")
                        continue
                    variant = list(stage1_dirs.keys())[0]
                else:
                    variant = stage1_variant

                d = stage1_dirs.get(variant)
                if d is None:
                    print(f"  [skip] sparse_moe/{moe_name}: stage1 variant '{variant}' not found")
                    continue

                semantic_map = d / "semantic_id_mappings.json"
                purified_content = d / "item_purified_content.npy"
                purified_collab = d / "item_purified_collab.npy"

                if not semantic_map.exists():
                    print(f"  [skip] sparse_moe/{moe_name}: missing semantic_id_mappings.json")
                    continue

                output_dir = self.output_base / moe_name
                moe_extra = ["--fusion_gate_type", "moe"] + extra_args
                exp = RecExperiment(
                    name=moe_name,
                    stage1_dir=d,
                    output_dir=output_dir,
                    semantic_map=semantic_map,
                    purified_content=purified_content,
                    purified_collab=purified_collab,
                    purified_dim=128,
                    log_path=output_dir / "training.log",
                    extra_rec_args=moe_extra,
                )

                if (output_dir / "best_model.pt").exists():
                    exp.status = "skipped"
                    log_m = parse_rec_metrics(self._resolve_log_path(exp))
                    for k, v in log_m.items():
                        setattr(exp, k, v)

                self.experiments.append(exp)

    def launch(self, exp: RecExperiment, gpu: int):
        """Launch recommender training on a specific GPU."""
        exp.output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TQDM_DISABLE"] = "1"

        # Auto-discover teacher_path from stage1 output or data directory
        teacher_path = exp.stage1_dir.parent / ".." / ".." / ".." / "dataset"
        # Resolve: hparam_stage1/<exp>/ -> hparam_stage1 -> prism_tokenizer -> output -> scripts -> PROJECT_ROOT
        data_teacher = self.project_root / "dataset" / DATASET_MAP[self.dataset] / "teacher_prototypes.npy"
        stage1_teacher = exp.stage1_dir / ".." / "teacher_prototypes.npy"
        if data_teacher.exists():
            resolved_teacher = str(data_teacher)
        elif stage1_teacher.exists():
            resolved_teacher = str(stage1_teacher.resolve())
        else:
            resolved_teacher = ""

        cmd = [
            sys.executable, "-m", REC_TRAIN_MODULE,
            "--config", self.dataset,
            "--output_dir", str(exp.output_dir),
            "--semantic_mapping_path", str(exp.semantic_map),
            "--purified_content_path", str(exp.purified_content),
            "--purified_collab_path", str(exp.purified_collab),
            "--purified_dim", str(exp.purified_dim),
            "--teacher_path", resolved_teacher,
            *BASE_REC_ARGS,
            *exp.extra_rec_args,
        ]
        if self.fast_dev_config is not None:
            cmd.extend(["--fast_dev_config", str(self.fast_dev_config)])

        # open log for header, then let recommender's FileHandler append
        with open(exp.log_path, "w") as hdr:
            hdr.write(f"# {'='*70}\n")
            hdr.write(f"# Stage 2 Rec Experiment: {exp.name}\n")
            hdr.write(f"# GPU: {gpu}  |  Stage1: {exp.stage1_dir}\n")
            hdr.write(f"# Started: {datetime.now().isoformat()}\n")
            hdr.write(f"# {'='*70}\n\n")

        # capture stdout+stderr to log file for crash diagnostics
        log_f = open(exp.log_path, "a")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                env=env, cwd=str(self.project_root))
        exp._log_f = log_f

        exp.status = "running"
        exp.gpu = gpu
        exp.proc = proc
        exp.started_at = time.time()
        self.running[gpu] = exp

    def _resolve_log_path(self, exp: RecExperiment) -> Path:
        training_log = exp.output_dir / "training.log"
        if training_log.exists():
            return training_log
        fallback_log = exp.output_dir / "stage1rec_training.log"
        return fallback_log if fallback_log.exists() else exp.log_path

    def poll(self):
        finished_gpus = []
        for gpu, exp in self.running.items():
            if exp.proc is None:
                continue
            ret = exp.proc.poll()
            if ret is not None:
                exp.finished_at = time.time()
                exp.status = "done" if ret == 0 else "failed"
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del self.running[gpu]
            self.gpu_queue.append(gpu)

        queued = [e for e in self.experiments if e.status == "queued"]
        while self.gpu_queue and queued:
            gpu = self.gpu_queue.pop(0)
            exp = queued.pop(0)
            self.launch(exp, gpu)

    def refresh_metrics(self):
        for exp in self.experiments:
            if exp.status == "running" and exp.log_path:
                m = parse_rec_metrics(self._resolve_log_path(exp))
                for k, v in m.items():
                    setattr(exp, k, v)

    def all_done(self) -> bool:
        return all(e.status in ("done", "failed", "skipped") for e in self.experiments)

    def terminate_all(self):
        for exp in self.experiments:
            if exp.proc and exp.proc.poll() is None:
                exp.proc.terminate()
                try:
                    exp.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    exp.proc.kill()
                exp.status = "failed"
                exp.finished_at = time.time()

    def run(self, refresh_interval: float = 3.0):
        self.discover_experiments()

        console.print(f"\n[bold magenta]PRISM Stage 2 — Recommender Batch Runner[/bold magenta]")
        console.print(f"Dataset: [bold]{self.dataset}[/bold]")
        console.print(f"GPUs: [bold]{self.gpus}[/bold]")
        console.print(f"Stage1 base: {self.stage1_base}")
        n_queued = sum(1 for e in self.experiments if e.status == "queued")
        n_skipped = sum(1 for e in self.experiments if e.status == "skipped")
        console.print(f"Experiments: [bold]{len(self.experiments)}[/bold] "
                      f"({n_queued} queued, {n_skipped} skipped)")
        console.print(f"Output: {self.output_base}\n")

        if all(e.status == "skipped" for e in self.experiments):
            console.print("[green]All experiments already completed![/green]")
            return

        interrupted = False

        def _sig_handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

        try:
            with Live(
                build_dashboard(self.experiments, self.gpus, self.running, self.start_time),
                console=console, refresh_per_second=1, screen=True, transient=True,
            ) as live:
                while not self.all_done() and not interrupted:
                    self.poll()
                    self.refresh_metrics()
                    live.update(
                        build_dashboard(self.experiments, self.gpus, self.running, self.start_time))
                    time.sleep(refresh_interval)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        if interrupted:
            console.print("\n[yellow]Interrupted. Stopping running experiments...[/yellow]")
            self.terminate_all()

        self.print_summary()

    def print_summary(self):
        console.print()
        n_done = sum(1 for e in self.experiments if e.status == "done")
        n_failed = sum(1 for e in self.experiments if e.status == "failed")
        n_skipped = sum(1 for e in self.experiments if e.status == "skipped")

        summary_table = Table(box=box.ROUNDED, title="Final Summary — Stage 2 Recommender (Test Set)")
        summary_table.add_column("Experiment")
        summary_table.add_column("Status")
        summary_table.add_column("S1 Val Loss")
        summary_table.add_column("S1 PreUniq%")
        summary_table.add_column("S1 Coll%")
        summary_table.add_column("S1 kNN")
        summary_table.add_column("S1 Seq")
        summary_table.add_column("Best")
        summary_table.add_column("Test R@10")
        summary_table.add_column("Test R@20")
        summary_table.add_column("Test N@10")
        summary_table.add_column("Test N@20")
        summary_table.add_column("Best Ep")
        summary_table.add_column("Time")

        for exp in self.experiments:
            icon = STATUS_ICONS.get(exp.status, "?")
            color = STATUS_COLORS.get(exp.status, "")
            dur = format_duration(
                (exp.finished_at - exp.started_at) if exp.finished_at and exp.started_at else None)
            best = f"{exp.best_metric:.4f}" if exp.best_metric else "--"
            s1_loss = f"{exp.stage1_val_loss:.4f}" if exp.stage1_val_loss else "--"
            s1_uniq = f"{exp.stage1_pre_unique_rate:.1%}" if exp.stage1_pre_unique_rate else "--"
            s1_coll = f"{exp.stage1_collision_item_rate:.1%}" if exp.stage1_collision_item_rate else "--"
            s1_knn = f"{exp.stage1_knn_l1:.2f}/{exp.stage1_knn_l12:.2f}" if exp.stage1_knn_l1 else "--"
            s1_seq = f"{exp.stage1_seq_l1:.2f}/{exp.stage1_seq_l2:.2f}" if exp.stage1_seq_l1 else "--"
            summary_table.add_row(
                exp.name,
                f"[{color}]{icon} {exp.status}[/{color}]",
                s1_loss, s1_uniq, s1_coll, s1_knn, s1_seq, best,
                f"{exp.test_recall10:.4f}" if exp.test_recall10 else "--",
                f"{exp.test_recall20:.4f}" if exp.test_recall20 else "--",
                f"{exp.test_ndcg10:.4f}" if exp.test_ndcg10 else "--",
                f"{exp.test_ndcg20:.4f}" if exp.test_ndcg20 else "--",
                str(exp.best_epoch) if exp.best_epoch else "--",
                dur,
            )

        console.print(summary_table)
        console.print(
            f"\n[bold]Total:[/bold] {len(self.experiments)}  |  "
            f"[green]{n_done} done[/green]  |  "
            f"[red]{n_failed} failed[/red]  |  "
            f"[dim]{n_skipped} skipped[/dim]"
        )

        if n_failed > 0:
            console.print("\n[red]Failed experiments:[/red]")
            for exp in self.experiments:
                if exp.status == "failed":
                    console.print(f"  - {exp.name}: see {exp.log_path}")

        # summary JSON
        summary = {}
        for exp in self.experiments:
            summary[exp.name] = {
                "status": exp.status,
                "stage1_val_loss": exp.stage1_val_loss,
                "stage1_pre_unique_rate": exp.stage1_pre_unique_rate,
                "stage1_collision_item_rate": exp.stage1_collision_item_rate,
                "stage1_knn_l1": exp.stage1_knn_l1,
                "stage1_knn_l12": exp.stage1_knn_l12,
                "stage1_prefix_l1": exp.stage1_prefix_l1,
                "stage1_prefix_l2": exp.stage1_prefix_l2,
                "stage1_prefix_l3": exp.stage1_prefix_l3,
                "stage1_code_usage_avg": exp.stage1_code_usage_avg,
                "stage1_seq_l1": exp.stage1_seq_l1,
                "stage1_seq_l2": exp.stage1_seq_l2,
                "stage1_seq_l3": exp.stage1_seq_l3,
                "stage1_seq_depth": exp.stage1_seq_depth,
                "best_metric": exp.best_metric if exp.best_metric else None,
                "test_recall10": exp.test_recall10 if exp.test_recall10 else None,
                "test_recall20": exp.test_recall20 if exp.test_recall20 else None,
                "test_ndcg10": exp.test_ndcg10 if exp.test_ndcg10 else None,
                "test_ndcg20": exp.test_ndcg20 if exp.test_ndcg20 else None,
                "best_epoch": exp.best_epoch if exp.best_epoch else None,
                "duration": format_duration(
                    (exp.finished_at - exp.started_at) if exp.finished_at and exp.started_at else None),
                "log": str(exp.log_path) if exp.log_path else None,
            }
        summary_path = self.output_base / "batch_summary.json"
        json.dump(summary, summary_path.open("w"), indent=2)
        console.print(f"\n[dim]Summary saved to {summary_path}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PRISM Stage 2 — Recommender Batch Runner")
    p.add_argument("dataset", nargs="?", default="beauty",
                   help="Dataset name (beauty, sports, toys, cds)")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU indices (default: all detected)")
    p.add_argument("--stage1-base", type=str, default=None,
                   help="Path to Stage 1 experiment output directory")
    p.add_argument("--experiments", type=str, default=None,
                   help="Comma-separated experiment names (default: full,ablate_both,ablate_cma,ablate_rgcp)")
    p.add_argument("--output-base", type=str, default=None,
                   help="Override Stage 2 output base directory")
    p.add_argument("--fast-dev-config", type=str, default=None,
                   help="Pass a fast-dev JSON config through to each recommender run")
    p.add_argument("--refresh", type=float, default=3.0,
                   help="Dashboard refresh interval in seconds")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = None
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(",")]
    stage1_base = Path(args.stage1_base) if args.stage1_base else None
    experiments = [x.strip() for x in args.experiments.split(",")] if args.experiments else None
    output_base = Path(args.output_base) if args.output_base else None
    fast_dev_config = Path(args.fast_dev_config) if args.fast_dev_config else None

    runner = RecBatchRunner(
        dataset=args.dataset,
        gpus=gpus,
        output_base=output_base,
        fast_dev_config=fast_dev_config,
        stage1_base=stage1_base,
        experiments=experiments,
    )
    runner.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
