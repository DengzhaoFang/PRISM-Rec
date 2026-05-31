#!/usr/bin/env python3
"""Batch runner for PRISM tokenizer (CMA+MCD+SACO)."""
import argparse, os, re, signal, subprocess, sys, time, json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "src/sid_tokenizer/prism/train_prism.py"

@dataclass
class Experiment:
    name: str
    extra_args: List[str]
    output_dir: Path
    status: str = "queued"
    gpu: int = -1
    proc = None; log_path = None; started_at = None; finished_at = None
    epoch: int = 0; train_loss: float = 0.0; val_loss: float = 0.0
    upr_loss: float = 0.0; commit_loss: float = 0.0
    z_norm: float = 0.0; z_total_var: float = 0.0; inter_cos: float = 0.0; neg_sim: float = 0.0
    codes_l1: int = 0; codes_l2: int = 0; codes_l3: int = 0
    best_val: float = float("inf"); _log_f = None

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
    ("Code Ablation", [
        ("cma_only",              ["--mcd", "off"]),
        ("cma_mcd",               []),
        ("cma_mcd_saco_c025",     ["--use_saco", "--lambda_sac", "0.1",
                                   "--commit_weight", "0.25"]),
        ("cma_mcd_saco_c00625",   ["--use_saco", "--lambda_sac", "0.1",
                                   "--commit_weight", "0.0625"]),
    ]),
]

_EPOCH_RE = re.compile(r"Epoch (\d+) Summary:")
_TRAIN_RE = re.compile(r"Train: Total=([\d.]+) UPR=([\d.]+) Commit=([\d.]+)")
_VAL_RE = re.compile(r"Val: Total=([\d.]+) UPR=([\d.]+)")
_BEST_RE = re.compile(r"New best recon: ([\d.]+)")
_CODES_RE = re.compile(r"L(\d): ([\d.]+)/256 codes")
_LATENT_RE = re.compile(r"\[2\] Latent z: norm=([\d.]+).*var=([\d.]+).*inter_cos=([\d.]+).*neg%=([\d.]+)")

def parse_metrics(log_path):
    m = {}
    if not log_path.exists(): return m
    try: text = log_path.read_text()
    except: return m
    epoch_starts = list(_EPOCH_RE.finditer(text))
    if not epoch_starts: return m
    block = text[epoch_starts[-1].start():]
    for pat, keys in [(_EPOCH_RE, [("epoch", int)]), (_TRAIN_RE, [("train_loss", float), ("upr_loss", float), ("commit_loss", float)]),
                       (_VAL_RE, [("val_loss", float), ("val_upr", float)])]:
        match = pat.search(block)
        if match:
            for i, (k, t) in enumerate(keys):
                try: m[k] = t(match.group(i+1))
                except: pass
    for match in _CODES_RE.finditer(block): m[f"codes_l{match.group(1)}"] = int(match.group(2))
    for pat, key in [(_BEST_RE, "best_val")]:
        bests = pat.findall(text)
        if bests: m[key] = min(float(b) for b in bests)
    match = _LATENT_RE.search(text)
    if match:
        for i, k in enumerate(["z_norm", "z_total_var", "inter_cos", "neg_sim"]):
            try: m[k] = float(match.group(i+1))
            except: pass
    return m

def detect_gpus(requested=None):
    try:
        import torch
        if torch.cuda.is_available():
            all_g = list(range(torch.cuda.device_count()))
            return [g for g in requested if g in all_g] if requested else all_g
    except: pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            gpus = [int(l.strip()) for l in r.stdout.strip().split("\n") if l.strip()]
            return [g for g in requested if g in gpus] if requested else gpus
    except: pass
    return []

console = Console()
STATUS_ICONS = {"queued":"⏳","running":"🔄","done":"✅","failed":"❌","skipped":"⏭️"}
STATUS_COLORS = {"queued":"dim","running":"yellow","done":"green","failed":"red","skipped":"dim"}

def fmt_dur(s):
    if s is None: return "--"
    m, sec = divmod(int(s), 60); h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{sec:02d}s"
    return f"{sec}s"

def build_dashboard(exps, gpus, running, start_time):
    layout = Layout(); layout.split(Layout(name="header",size=3), Layout(name="gpu_bar",size=2), Layout(name="body"), Layout(name="footer",size=3))
    n_queued = sum(1 for e in exps if e.status=="queued"); n_running = sum(1 for e in exps if e.status=="running")
    n_done = sum(1 for e in exps if e.status=="done"); n_failed = sum(1 for e in exps if e.status=="failed")
    elapsed = fmt_dur(time.time()-start_time)
    ht = Text(); ht.append("PRISM Stage 1 — Batch Runner", style="bold cyan")
    ht.append(f"\n{elapsed} elapsed | {n_running} running {n_queued} queued {n_done} done", style="dim")
    layout["header"].update(Panel(ht, box=box.ROUNDED))
    gt = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    gt.add_column("gpu", style="dim", width=6); gt.add_column("experiment")
    for gid in gpus:
        exp = running.get(gid)
        gt.add_row(f"GPU {gid}", f"[yellow]{exp.name}[/yellow] Epoch {exp.epoch}" if exp else "[dim]idle[/dim]")
    layout["gpu_bar"].update(Panel(gt, box=box.ROUNDED, title="GPUs"))
    et = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0,1), expand=True)
    for h in ["#","Experiment","Status","Epoch","Val","Best","UPR","z_norm","inter_cos","GPU","Time"]:
        et.add_column(h, width=max(4,min(len(h)+2,12)), justify="right" if h not in ["Experiment","Status"] else "left", no_wrap=True)
    for i, exp in enumerate(exps):
        icon = STATUS_ICONS.get(exp.status,"?"); color = STATUS_COLORS.get(exp.status,"")
        dur = fmt_dur(time.time()-exp.started_at) if exp.status=="running" else (fmt_dur(exp.finished_at-exp.started_at) if exp.finished_at and exp.started_at else "--")
        gpu_label = str(exp.gpu) if exp.gpu>=0 else "--"
        def _f(v, d=4): return f"{v:.{d}f}" if v else "--"
        et.add_row(str(i+1), exp.name, f"[{color}]{icon} {exp.status}[/{color}]", str(exp.epoch) if exp.epoch else "--",
                   _f(exp.val_loss), _f(exp.best_val) if exp.best_val<float("inf") else "--", _f(exp.upr_loss),
                   _f(exp.z_norm,2) if exp.z_norm else "--", _f(exp.inter_cos,4) if exp.inter_cos else "--", gpu_label, dur)
    layout["body"].update(et)
    layout["footer"].update(Panel(Text("Ctrl+C to stop", style="dim"), box=box.ROUNDED))
    return layout

class BatchRunner:
    def __init__(self, dataset, gpus=None, output_base=None, experiments=None):
        self.dataset = dataset; self.gpus = detect_gpus(gpus)
        if not self.gpus: print("ERROR: No GPUs!"); sys.exit(1)
        if dataset not in DATASET_MAP: print(f"Unknown dataset: {dataset}"); sys.exit(1)
        self.data_path = str((PROJECT_ROOT/"dataset"/DATASET_MAP[dataset]).resolve())
        self.output_base = output_base or (PROJECT_ROOT/"scripts/output/prism_tokenizer"/dataset/"hparam_stage1")
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.experiments: List[Experiment] = []; self.gpu_queue = list(self.gpus)
        self.running = {}; self.start_time = time.time()
        self.target_experiments = experiments or []

    def enqueue(self):
        for gname, exp_list in EXPERIMENT_GROUPS:
            for name, extra_args in exp_list:
                od = self.output_base / name
                exp = Experiment(name=name, extra_args=extra_args, output_dir=od, log_path=od/"training.log")
                if (od/"best_model.pt").exists(): exp.status = "skipped"
                self.experiments.append(exp)

    def launch(self, exp, gpu):
        exp.output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--data_path", self.data_path, "--output_dir", str(exp.output_dir), "--device", "cuda", *BASE_TRAIN_ARGS, *exp.extra_args]
        with open(exp.log_path, "w") as hdr:
            hdr.write(f"# Experiment: {exp.name}\n# GPU: {gpu}\n# Started: {datetime.now().isoformat()}\n# {' '.join(cmd)}\n\n")
        exp._log_f = open(exp.log_path, "a")
        exp.proc = subprocess.Popen(cmd, stdout=exp._log_f, stderr=subprocess.STDOUT, env=env, cwd=str(TRAIN_SCRIPT.parent))
        exp.status = "running"; exp.gpu = gpu; exp.started_at = time.time(); self.running[gpu] = exp

    def poll(self):
        finished = []
        for gpu, exp in self.running.items():
            if exp.proc and exp.proc.poll() is not None:
                exp.finished_at = time.time(); exp.status = "done" if exp.proc.returncode==0 else "failed"
                finished.append(gpu)
        for gpu in finished: del self.running[gpu]; self.gpu_queue.append(gpu)
        queued = [e for e in self.experiments if e.status=="queued"]
        while self.gpu_queue and queued:
            self.launch(queued.pop(0), self.gpu_queue.pop(0))

    def refresh(self):
        for exp in self.experiments:
            if exp.status=="running" and exp.log_path:
                m = parse_metrics(exp.log_path)
                for k, v in m.items():
                    if hasattr(exp, k): setattr(exp, k, v)

    def all_done(self): return all(e.status in ("done","failed","skipped") for e in self.experiments)

    def terminate(self):
        for exp in self.experiments:
            if exp.proc and exp.proc.poll() is None:
                exp.proc.terminate()
                try: exp.proc.wait(timeout=5)
                except: exp.proc.kill()
                exp.status="failed"; exp.finished_at=time.time()

    def run(self, refresh_interval=3.0):
        self.enqueue()
        console.print(f"\n[bold cyan]PRISM Stage 1 — Batch Runner[/bold cyan]")
        console.print(f"Dataset: [bold]{self.dataset}[/bold] | GPUs: {self.gpus} | Experiments: {len(self.experiments)}")
        if all(e.status=="skipped" for e in self.experiments): console.print("[green]All done![/green]"); return
        interrupted = False
        def handler(s,f): nonlocal interrupted; interrupted = True
        prev_sig = signal.signal(signal.SIGINT, handler)
        try:
            with Live(build_dashboard(self.experiments, self.gpus, self.running, self.start_time), console=console, refresh_per_second=1, screen=True, transient=True) as live:
                while not self.all_done() and not interrupted:
                    self.poll(); self.refresh()
                    live.update(build_dashboard(self.experiments, self.gpus, self.running, self.start_time))
                    time.sleep(refresh_interval)
        finally: signal.signal(signal.SIGINT, prev_sig)
        if interrupted: console.print("\n[yellow]Interrupted![/yellow]"); self.terminate()
        self.print_summary()

    def print_summary(self):
        console.print()
        st = Table(box=box.ROUNDED, title="Final Summary")
        for h in ["Experiment","Status","Best Val","UPR","z_norm","inter_cos","neg_sim%","Epochs","Time"]:
            st.add_column(h)
        for exp in self.experiments:
            icon = STATUS_ICONS.get(exp.status,"?"); color = STATUS_COLORS.get(exp.status,"")
            dur = fmt_dur((exp.finished_at-exp.started_at) if exp.finished_at and exp.started_at else None)
            best = f"{exp.best_val:.4f}" if exp.best_val<float("inf") else "--"
            st.add_row(exp.name, f"[{color}]{icon} {exp.status}[/{color}]", best,
                       f"{exp.upr_loss:.4f}" if exp.upr_loss else "--",
                       f"{exp.z_norm:.2f}" if exp.z_norm else "--",
                       f"{exp.inter_cos:.4f}" if exp.inter_cos else "--",
                       f"{exp.neg_sim*100:.1f}" if exp.neg_sim else "--",
                       str(exp.epoch) if exp.epoch else "--", dur)
        console.print(st)
        console.print(f"\n[bold]Total: {len(self.experiments)} | [green]{sum(1 for e in self.experiments if e.status=='done')} done[/green] | [red]{sum(1 for e in self.experiments if e.status=='failed')} failed[/red]")
        summary = {}
        for exp in self.experiments:
            summary[exp.name] = {"status":exp.status, "best_val":exp.best_val if exp.best_val<float("inf") else None,
                "upr_loss":exp.upr_loss, "epoch":exp.epoch, "z_norm":exp.z_norm, "z_total_var":exp.z_total_var,
                "inter_cos":exp.inter_cos, "neg_sim":exp.neg_sim, "duration":fmt_dur((exp.finished_at-exp.started_at) if exp.finished_at and exp.started_at else None)}
        sp = self.output_base/"batch_summary.json"
        sp.write_text(json.dumps(summary, indent=2))
        console.print(f"\n[dim]Summary saved to {sp}[/dim]")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", default="beauty")
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--output-base", type=str, default=None)
    p.add_argument("--refresh", type=float, default=3.0)
    args = p.parse_args()
    gpus = [int(x.strip()) for x in args.gpus.split(",")] if args.gpus else None
    output_base = Path(args.output_base) if args.output_base else None
    BatchRunner(dataset=args.dataset, gpus=gpus, output_base=output_base).run(refresh_interval=args.refresh)

if __name__ == "__main__":
    main()
