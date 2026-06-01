#!/usr/bin/env python3
"""
PRISM Stage 1 — Batch Experiment Runner with GPU Scheduling & Live Dashboard

Features:
- Dynamic GPU scheduling: runs N experiments in parallel across available GPUs
- Live terminal dashboard showing experiment status and key metrics
- Auto-skip experiments that already have best_model.pt
- Detailed logs written to per-experiment log files

Usage:
  python scripts/prism/batch/stage1.py [DATASET] [--gpus 0,1,2,3]
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
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
TRAIN_SCRIPT = PROJECT_ROOT / "src/sid_tokenizer/prism/train_prism.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experiment:
    name: str
    extra_args: List[str]
    output_dir: Path
    status: str = "queued"  # queued | running | done | failed | skipped
    gpu: int = -1
    proc: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # metrics extracted from log
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    upr_loss: float = 0.0
    commit_loss: float = 0.0
    # latent space metrics
    z_norm: Optional[float] = None
    z_total_var: Optional[float] = None
    inter_cos: Optional[float] = None
    neg_sim: Optional[float] = None
    high_sim: Optional[float] = None
    best_val: float = float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Definitions (mirrors hparam_sensitivity_stage1.sh)
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_MAP = {
    "beauty": "Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
    "sports": "Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports",
    "toys": "Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys",
    "cds": "Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
}

BASE_TRAIN_ARGS = [
    "--n_layers", "3", "--n_embed_per_layer", "256,256,256", "--latent_dim", "32",
    "--content_dim", "768", "--collab_dim", "64",
    "--epochs", "500", "--batch_size", "512", "--learning_rate", "1e-4",
    "--weight_decay", "1e-4", "--grad_clip", "1.0",
    "--commit_weight", "0.0625",
    "--use_ema", "--ema_decay", "0.99", "--quantize_mode", "rotation",
    "--use_scheduler", "--scheduler_type", "warmup_cosine", "--warmup_ratio", "0.1",
    "--early_stop_patience", "30", "--early_stop_min_delta", "1e-5",
    "--early_stop_cooldown", "3", "--early_stop_warmup_epochs", "5",
    "--perplexity_collapse_ratio", "0.35", "--perplexity_collapse_patience", "3",
    "--no_hierarchical_kmeans_init", "--kmeans_init_samples", "8192",
    "--save_every", "50", "--num_workers", "4", "--log_level", "INFO",
    "--ide", "on", "--ide_dim", "128",
]

EXPERIMENT_GROUPS = [
    ("Module Ablation", [
        ("cma_only",              ["--mcd", "off"]),
        ("cma_mcd",               []),
        ("cma_mcd_saco_c025",     ["--use_saco", "--lambda_sac", "0.1",
                                   "--commit_weight", "0.25"]),
        ("cma_mcd_saco_c00625",   ["--use_saco", "--lambda_sac", "0.1",
                                   "--commit_weight", "0.0625"]),
    ]),
]

EXPERIMENT_NAME_ALIASES = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Log Parser
# ═══════════════════════════════════════════════════════════════════════════════

_EPOCH_RE = re.compile(r"Epoch (\d+) Summary:")
_TRAIN_RE = re.compile(r"Total Loss:\s+([\d.]+)")
_UPR_RE = re.compile(r"UPR Loss:\s+([\d.]+)")
_PERPLEXITY_RE = re.compile(r"Perplexity Layer (\d): ([\d.]+)")
_BEST_RE = re.compile(r"New best loss: ([\d.]+)")
# Match OLD log format: "[2] Latent z: norm=5.35±0.29 var=15.40 ... inter_cos=0.4598 neg%=1.0"
_LATENT_RE = re.compile(r"\[2\] Latent z: norm=([\d.]+).*var=([\d.]+).*inter_cos=([\d.]+).*neg%=([\d.]+)")


def parse_metrics_from_log(log_path: Path) -> Dict:
    """Scan the log file and extract the latest epoch metrics."""
    metrics = {}
    if not log_path.exists():
        return metrics
    try:
        text = log_path.read_text()
    except Exception:
        return metrics

    # find all epoch blocks, keep the last one
    epoch_starts = list(_EPOCH_RE.finditer(text))
    if not epoch_starts:
        return metrics

    # last epoch block
    last_start = epoch_starts[-1].start()
    block = text[last_start:]

    m = _EPOCH_RE.search(block)
    if m:
        metrics["epoch"] = int(m.group(1))

    m = _TRAIN_RE.search(block)
    if m:
        metrics["train_loss"] = float(m.group(1))
    m = _UPR_RE.search(block)
    if m:
        metrics["upr_loss"] = float(m.group(1))

    for m in _PERPLEXITY_RE.finditer(block):
        metrics[f"perp_l{m.group(1)}"] = float(m.group(2))

    # latent space analysis (post-training, full log scan)
    match = _LATENT_RE.search(text)
    if match:
        for i, key in enumerate(["z_norm", "z_total_var", "inter_cos", "neg_sim"]):
            try: metrics[key] = float(match.group(i+1))
            except: pass

    # find best val so far
    bests = _BEST_RE.findall(block)
    if bests:
        metrics["best_val"] = float(bests[-1])
    else:
        # scan whole log for best
        all_bests = _BEST_RE.findall(text)
        if all_bests:
            metrics["best_val"] = min(float(b) for b in all_bests)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Detection & Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpus(requested: Optional[List[int]] = None) -> List[int]:
    """Detect available GPUs. Returns list of GPU indices."""
    try:
        import torch
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if requested is not None:
                return [g for g in requested if g in all_gpus]
            return all_gpus
    except ImportError:
        pass

    # fallback: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
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

STATUS_ICONS = {
    "queued":  "⏳",
    "running": "🔄",
    "done":    "✅",
    "failed":  "❌",
    "skipped": "⏭️",
}

STATUS_COLORS = {
    "queued":  "dim",
    "running": "yellow",
    "done":    "green",
    "failed":  "red",
    "skipped": "dim",
}


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


def build_dashboard(exps: List[Experiment], gpus: List[int],
                    running_exp_per_gpu: Dict[int, Optional[Experiment]],
                    start_time: float) -> Layout:
    """Build the rich layout for the live dashboard."""

    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="gpu_bar", size=2),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # ── header ──
    n_queued = sum(1 for e in exps if e.status == "queued")
    n_running = sum(1 for e in exps if e.status == "running")
    n_done = sum(1 for e in exps if e.status == "done")
    n_failed = sum(1 for e in exps if e.status == "failed")
    n_skipped = sum(1 for e in exps if e.status == "skipped")
    elapsed = format_duration(time.time() - start_time)

    header_text = Text()
    header_text.append("PRISM Stage 1 — Batch Runner", style="bold cyan")
    header_text.append(f"\n{elapsed} elapsed  |  ", style="dim")
    header_text.append(f"{n_running} running", style="yellow")
    header_text.append(f"  {n_queued} queued", style="dim")
    header_text.append(f"  {n_done} done", style="green")
    if n_failed:
        header_text.append(f"  {n_failed} failed", style="red")
    if n_skipped:
        header_text.append(f"  {n_skipped} skipped", style="dim")

    layout["header"].update(Panel(header_text, box=box.ROUNDED))

    # ── GPU bar ──
    gpu_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    gpu_table.add_column("gpu", style="dim", width=6)
    gpu_table.add_column("experiment")
    for gpu_id in gpus:
        exp = running_exp_per_gpu.get(gpu_id)
        if exp:
            status = f"[yellow]{exp.name}[/yellow] Epoch {exp.epoch}"
        else:
            status = "[dim]idle[/dim]"
        gpu_table.add_row(f"GPU {gpu_id}", status)
    layout["gpu_bar"].update(Panel(gpu_table, box=box.ROUNDED, title="GPUs"))

    # ── body: experiment table ──
    exp_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        expand=True,
    )
    exp_table.add_column("#", width=3, style="dim", no_wrap=True)
    exp_table.add_column("Experiment", min_width=12, max_width=14, no_wrap=True, overflow="ellipsis")
    exp_table.add_column("Status", width=10, no_wrap=True)
    exp_table.add_column("Epoch", width=6, justify="right", no_wrap=True)
    exp_table.add_column("Val", width=8, justify="right", no_wrap=True)
    exp_table.add_column("Best", width=8, justify="right", no_wrap=True)
    exp_table.add_column("UPR", width=8, justify="right", no_wrap=True)
    exp_table.add_column("z_norm", width=8, justify="right", no_wrap=True)
    exp_table.add_column("inter_cos", width=9, justify="right", no_wrap=True)
    exp_table.add_column("GPU", width=4, justify="center", no_wrap=True)
    exp_table.add_column("Time", width=8, no_wrap=True)

    for i, exp in enumerate(exps):
        icon = STATUS_ICONS.get(exp.status, "?")
        color = STATUS_COLORS.get(exp.status, "")

        if exp.status == "running":
            dur = format_duration(time.time() - exp.started_at) if exp.started_at else "--"
        elif exp.finished_at and exp.started_at:
            dur = format_duration(exp.finished_at - exp.started_at)
        else:
            dur = "--"

        gpu_label = str(exp.gpu) if exp.gpu >= 0 else "--"

        def _fmt(v, dec=4):
            return f"{v:.{dec}f}" if v else "--"

        exp_table.add_row(
            str(i + 1),
            exp.name,
            f"[{color}]{icon} {exp.status}[/{color}]",
            str(exp.epoch) if exp.epoch else "--",
            _fmt(exp.val_loss),
            _fmt(exp.best_val) if exp.best_val < float("inf") else "--",
            _fmt(exp.upr_loss),
            _fmt(exp.z_norm, dec=2) if exp.z_norm is not None else "--",
            _fmt(exp.inter_cos, dec=4) if exp.inter_cos is not None else "--",
            gpu_label,
            dur,
        )

    layout["body"].update(exp_table)

    # ── footer ──
    footer_text = Text()
    footer_text.append("Detailed logs: ", style="dim")
    footer_text.append("{output_base}/<exp_name>/training.log", style="dim italic")
    footer_text.append("  │  ", style="dim")
    footer_text.append("Ctrl+C", style="bold")
    footer_text.append(" to stop gracefully", style="dim")
    layout["footer"].update(Panel(footer_text, box=box.ROUNDED))

    return layout


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class BatchRunner:
    def __init__(self, dataset: str, gpus: Optional[List[int]] = None,
                 output_base: Optional[Path] = None):
        self.dataset = dataset
        self.gpus = detect_gpus(gpus)
        if not self.gpus:
            print("ERROR: No GPUs detected!")
            sys.exit(1)

        if dataset not in DATASET_MAP:
            print(f"ERROR: Unknown dataset '{dataset}'. Choose from {list(DATASET_MAP.keys())}")
            sys.exit(1)

        data_dir = PROJECT_ROOT / "dataset" / DATASET_MAP[dataset]
        self.data_path = str(data_dir.resolve())
        if output_base is None:
            resolved_output_base = PROJECT_ROOT / "scripts/output/prism_tokenizer" / dataset / "hparam_stage1"
        else:
            resolved_output_base = output_base if output_base.is_absolute() else (PROJECT_ROOT / output_base)
        self.output_base = resolved_output_base.resolve()

        self.experiments: List[Experiment] = []
        self.gpu_queue: List[int] = list(self.gpus)
        self.running: Dict[int, Experiment] = {}  # gpu_id -> Experiment
        self.start_time = time.time()

    def enqueue_experiments(self):
        for group_name, exp_list in EXPERIMENT_GROUPS:
            for name, extra_args in exp_list:
                output_dir = self.output_base / name
                exp = Experiment(
                    name=name,
                    extra_args=extra_args,
                    output_dir=output_dir,
                    log_path=output_dir / "training.log",
                )
                # check if already done
                if (output_dir / "best_model.pt").exists():
                    exp.status = "skipped"
                self.experiments.append(exp)

    def launch(self, exp: Experiment, gpu: int):
        """Launch an experiment on a specific GPU."""
        output_dir = exp.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TQDM_DISABLE"] = "1"

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data_path", self.data_path,
            "--output_dir", str(output_dir),
            "--device", "cuda",
            *BASE_TRAIN_ARGS,
            *exp.extra_args,
        ]

        # write header, then let tokenizer's FileHandler append cleanly
        with open(exp.log_path, "w") as hdr:
            hdr.write(f"# {'='*70}\n")
            hdr.write(f"# Stage 1 Experiment: {exp.name}\n")
            hdr.write(f"# GPU: {gpu}\n")
            hdr.write(f"# Started: {datetime.now().isoformat()}\n")
            hdr.write(f"# Command: {' '.join(cmd)}\n")
            hdr.write(f"# {'='*70}\n\n")

        log_f = open(exp.log_path, "a")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                env=env, cwd=str(TRAIN_SCRIPT.parent))
        exp._log_f = log_f  # keep alive for poll/terminate

        exp.status = "running"
        exp.gpu = gpu
        exp.proc = proc
        exp.started_at = time.time()
        self.running[gpu] = exp

    def poll(self):
        """Check if any running experiment has finished, launch queued ones."""
        finished_gpus = []
        for gpu, exp in self.running.items():
            if exp.proc is None:
                continue
            ret = exp.proc.poll()
            if ret is not None:
                # process finished
                exp.finished_at = time.time()
                exp.status = "done" if ret == 0 else "failed"
                finished_gpus.append(gpu)

        # free up finished GPUs
        for gpu in finished_gpus:
            del self.running[gpu]
            self.gpu_queue.append(gpu)

        # launch queued experiments on available GPUs
        queued = [e for e in self.experiments if e.status == "queued"]
        while self.gpu_queue and queued:
            gpu = self.gpu_queue.pop(0)
            exp = queued.pop(0)
            self.launch(exp, gpu)

    def refresh_metrics(self):
        """Read each running experiment's log and update its metrics."""
        for exp in self.experiments:
            if exp.status == "running" and exp.log_path:
                m = parse_metrics_from_log(exp.log_path)
                for k, v in m.items():
                    if hasattr(exp, k):
                        setattr(exp, k, v)

    def all_done(self) -> bool:
        return all(e.status in ("done", "failed", "skipped") for e in self.experiments)

    def terminate_all(self):
        """Kill all running experiment processes."""
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
        self.enqueue_experiments()

        console.print(f"\n[bold cyan]PRISM Stage 1 — Batch Runner[/bold cyan]")
        console.print(f"Dataset: [bold]{self.dataset}[/bold]")
        console.print(f"GPUs: [bold]{self.gpus}[/bold]")
        console.print(f"Experiments: [bold]{len(self.experiments)}[/bold] "
                      f"({sum(1 for e in self.experiments if e.status=='queued')} queued, "
                      f"{sum(1 for e in self.experiments if e.status=='skipped')} skipped)")
        console.print(f"Output: {self.output_base}\n")

        if all(e.status == "skipped" for e in self.experiments):
            console.print("[green]All experiments already completed![/green]")
            return

        # graceful shutdown on Ctrl+C
        interrupted = False

        def _sig_handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

        try:
            with Live(
                build_dashboard(self.experiments, self.gpus, self.running, self.start_time),
                console=console,
                refresh_per_second=1,
                screen=True,
                transient=True,
            ) as live:
                while not self.all_done() and not interrupted:
                    self.poll()
                    self.refresh_metrics()
                    live.update(
                        build_dashboard(self.experiments, self.gpus, self.running, self.start_time)
                    )
                    time.sleep(refresh_interval)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        if interrupted:
            console.print("\n[yellow]Interrupted. Stopping running experiments...[/yellow]")
            self.terminate_all()

        # final summary
        self.print_summary()

    def print_summary(self):
        console.print()
        n_done = sum(1 for e in self.experiments if e.status == "done")
        n_failed = sum(1 for e in self.experiments if e.status == "failed")
        n_skipped = sum(1 for e in self.experiments if e.status == "skipped")

        summary_table = Table(box=box.ROUNDED, title="Final Summary")
        summary_table.add_column("Experiment")
        summary_table.add_column("Status")
        summary_table.add_column("Best Val")
        summary_table.add_column("UPR")
        summary_table.add_column("z_norm")
        summary_table.add_column("inter_cos")
        summary_table.add_column("neg_sim%")
        summary_table.add_column("Epochs")
        summary_table.add_column("Time")

        for exp in self.experiments:
            icon = STATUS_ICONS.get(exp.status, "?")
            color = STATUS_COLORS.get(exp.status, "")
            dur = format_duration(
                (exp.finished_at - exp.started_at) if exp.finished_at and exp.started_at else None
            )
            best = f"{exp.best_val:.4f}" if exp.best_val < float("inf") else "--"
            z_norm_str = f"{exp.z_norm:.2f}" if exp.z_norm is not None else "--"
            inter_cos_str = f"{exp.inter_cos:.4f}" if exp.inter_cos is not None else "--"
            neg_sim_str = f"{exp.neg_sim*100:.1f}" if exp.neg_sim is not None else "--"
            summary_table.add_row(
                exp.name,
                f"[{color}]{icon} {exp.status}[/{color}]",
                best,
                f"{exp.upr_loss:.4f}" if exp.upr_loss else "--",
                z_norm_str,
                inter_cos_str,
                neg_sim_str,
                str(exp.epoch) if exp.epoch else "--",
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

        # write summary JSON
        summary = {}
        for exp in self.experiments:
            summary[exp.name] = {
                "status": exp.status,
                "best_val": exp.best_val if exp.best_val < float("inf") else None,
                "upr_loss": exp.upr_loss if exp.upr_loss else None,
                "epoch": exp.epoch,
                "z_norm": exp.z_norm if exp.z_norm is not None else None,
                "z_total_var": exp.z_total_var if exp.z_total_var is not None else None,
                "inter_cos": exp.inter_cos if exp.inter_cos is not None else None,
                "neg_sim_pct": round(exp.neg_sim * 100, 1) if exp.neg_sim is not None else None,
                "high_sim_pct": round(exp.high_sim * 100, 1) if exp.high_sim is not None else None,
                "duration": format_duration(
                    (exp.finished_at - exp.started_at) if exp.finished_at and exp.started_at else None
                ),
                "log": str(exp.log_path) if exp.log_path else None,
            }
        summary_path = self.output_base / "batch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        summary_path.write_text(json.dumps(summary, indent=2))
        console.print(f"\n[dim]Summary saved to {summary_path}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PRISM Stage 1 — Batch Experiment Runner")
    p.add_argument("dataset", nargs="?", default="beauty",
                   help="Dataset name (beauty, sports, toys, cds)")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU indices (default: all detected)")
    p.add_argument("--output-base", type=str, default=None,
                   help="Override output base directory")
    p.add_argument("--refresh", type=float, default=3.0,
                   help="Dashboard refresh interval in seconds")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = None
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(",")]
    output_base = Path(args.output_base) if args.output_base else None

    runner = BatchRunner(dataset=args.dataset, gpus=gpus, output_base=output_base)
    runner.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
