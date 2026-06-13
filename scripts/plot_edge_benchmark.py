"""Render the six edge-benchmark charts from reports/edge_benchmark/metrics.json.

Standalone: `uv run python scripts/plot_edge_benchmark.py [metrics.json] [out_dir]`.
Also called by scripts/benchmark_edge.py at the end of a run. Reads only the
JSON — no re-benchmarking needed to regenerate charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROXY = "Pi-class proxy: 1 CPU core, 512 MB — not measured on physical Pi hardware."
HEADLINE = "dev-1core"           # the label used for single-environment charts
SOURCE_ORDER = ["metropt", "hydraulic", "cmapss"]
SOURCE_LABEL = {"metropt": "Metro.PT\ncompressor", "hydraulic": "Hydraulic\nrig", "cmapss": "CMAPSS\nturbofan"}
ACCENT = "#0e7490"
DPI = 150


def _runs(doc):
    return doc.get("runs", {})


def _headline_label(doc):
    runs = _runs(doc)
    return HEADLINE if HEADLINE in runs else (next(iter(runs)) if runs else None)


def _sources(run):
    srcs = run.get("sources", {})
    return [s for s in SOURCE_ORDER if s in srcs] + [s for s in srcs if s not in SOURCE_ORDER]


def _footnote(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=6.3, color="#555")


# ───────────────────────── 1. latency CDF ────────────────────────────────

def chart_latency_cdf(doc, out):
    runs = _runs(doc)
    all_sources = []
    for run in runs.values():
        for s in _sources(run):
            if s not in all_sources:
                all_sources.append(s)
    if not all_sources:
        return
    n = len(all_sources)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    y = np.arange(0, 101) / 100.0
    for ax, src in zip(axes[0], all_sources):
        for label, run in runs.items():
            ent = run.get("sources", {}).get(src)
            if not ent:
                continue
            cdf = ent["latency_ms"].get("total_cdf_ms")
            if not cdf:
                continue
            ax.plot(cdf, y, label=label, linewidth=1.8)
            p99 = ent["latency_ms"]["total"]["p99"]
            ax.axvline(p99, color="#bbb", linestyle=":", linewidth=0.8)
        ax.set_title(SOURCE_LABEL.get(src, src).replace("\n", " "), fontsize=10)
        ax.set_xlabel("per-window latency (ms)")
        ax.set_ylabel("cumulative fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="environment")
    fig.suptitle("Per-window inference latency — CDF (p99 marked)", fontsize=12)
    _footnote(fig, PROXY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "latency_cdf.png", dpi=DPI)
    plt.close(fig)


# ───────────────────────── 2. pipeline breakdown ─────────────────────────

def chart_pipeline_breakdown(doc, out):
    label = _headline_label(doc)
    run = _runs(doc).get(label, {})
    srcs = _sources(run)
    if not srcs:
        return
    stages = ["preprocess", "forward", "attribution", "latent_match"]
    colors = ["#94a3b8", ACCENT, "#f59e0b", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(srcs) + 1.8))
    ypos = np.arange(len(srcs))
    left = np.zeros(len(srcs))
    for stage, color in zip(stages, colors):
        vals = [run["sources"][s]["latency_ms"][stage]["mean"] for s in srcs]
        ax.barh(ypos, vals, left=left, color=color, label=stage, height=0.55)
        left += np.array(vals)
    for i, s in enumerate(srcs):
        ax.text(left[i], ypos[i], f"  {left[i]:.2f} ms", va="center", fontsize=9, color="#333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([SOURCE_LABEL.get(s, s).replace("\n", " ") for s in srcs])
    ax.set_xlabel("mean latency per window (ms)")
    ax.set_title(f"Inference pipeline stage breakdown ({label})")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    _footnote(fig, PROXY + "   ·   includes per-feature attribution + Layer-3 latent match")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "pipeline_breakdown.png", dpi=DPI)
    plt.close(fig)


# ───────────────────────── 3. headroom ───────────────────────────────────

def chart_headroom(doc, out):
    label = _headline_label(doc)
    run = _runs(doc).get(label, {})
    srcs = _sources(run)
    if not srcs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(srcs))
    w = 0.38
    ach = [run["sources"][s]["throughput_wps"] for s in srcs]
    req = [run["sources"][s]["required_wps"] for s in srcs]
    ax.bar(x - w / 2, ach, w, color=ACCENT, label="achieved (windows/s)")
    ax.bar(x + w / 2, req, w, color="#cbd5e1", label="required by asset (windows/s)")
    ax.set_yscale("log")
    for i, s in enumerate(srcs):
        ratio = run["sources"][s]["headroom"]
        ax.text(x[i], ach[i] * 1.3, f"~{ratio:,.0f}× real-time", ha="center", fontsize=9, color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in srcs])
    ax.set_ylabel("windows / second (log scale)")
    ax.set_title(f"Achieved throughput vs the rate the asset actually needs ({label})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    _footnote(fig, PROXY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "headroom.png", dpi=DPI)
    plt.close(fig)


# ───────────────────────── 4. assets per device (pitch) ──────────────────

def chart_assets_per_device(doc, out):
    label = _headline_label(doc)
    run = _runs(doc).get(label, {})
    srcs = _sources(run)
    if not srcs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(srcs))
    vals = [run["sources"][s]["assets_per_core"] for s in srcs]
    bars = ax.bar(x, vals, color=ACCENT, width=0.55)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in srcs])
    ax.set_ylabel("assets one 1-core device can monitor (log)")
    ax.set_title("How many assets fit on a single cheap edge core")
    ax.grid(axis="y", alpha=0.2, which="both")
    _footnote(fig, PROXY + "   ·   conservative: full fp32 pipeline, one window at a time")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "assets_per_device.png", dpi=DPI)
    plt.close(fig)


# ───────────────────────── 5. footprint ──────────────────────────────────

def chart_footprint(doc, out):
    label = _headline_label(doc)
    run = _runs(doc).get(label, {})
    srcs = _sources(run)
    if not srcs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(srcs))
    model_mb = [run["sources"][s]["model"]["fp32_bytes"] / 1e6 for s in srcs]
    proc_rss = run.get("inference_process_rss_mb") or 0.0
    torch_base = run.get("torch_baseline_rss_mb") or 0.0
    bars = ax.bar(x, model_mb, 0.5, color=ACCENT, label="model fp32 (the shipped artifact)")
    for b, v in zip(bars, model_mb):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v*1000:.0f} KB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(512, color="#dc2626", linestyle="--", linewidth=1.5, label="512 MB budget")
    if proc_rss:
        ax.axhline(proc_rss, color="#f59e0b", linestyle="-", linewidth=1.6,
                   label=f"fp32 PyTorch inference process: {proc_rss:.0f} MB")
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in srcs])
    ax.set_ylabel("megabytes")
    ax.set_ylim(0, max(560, proc_rss * 1.12))
    ax.set_title(f"Memory footprint vs a 512 MB edge budget ({label})")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.25)
    _footnote(fig, PROXY + f"   ·   RSS = x86 PyTorch runtime (~{torch_base:.0f} MB), not the model; ONNX/tflite Pi build is lighter (deferred)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "footprint.png", dpi=DPI)
    plt.close(fig)


# ───────────────────────── 6. startup ────────────────────────────────────

def chart_startup(doc, out):
    label = _headline_label(doc)
    run = _runs(doc).get(label, {})
    srcs = _sources(run)
    if not srcs:
        return
    # Use the first source's startup numbers (process-level, same for the run).
    ent = run["sources"][srcs[0]]
    cold = ent.get("startup_cold_s") or 75.0
    warm = ent.get("startup_warm_s") or 0.0
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(["cold start\n(calibrate + train)", "warm start\n(load showcase model)"],
                  [cold, warm], color=["#cbd5e1", ACCENT], width=0.55)
    for b, v in zip(bars, [cold, warm]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ratio = cold / warm if warm else 0
    ax.set_ylabel("seconds to first scored window")
    ax.set_title("Demo start: cold vs warm")
    if ratio:
        ax.text(0.5, 0.92, f"~{ratio:.0f}× faster demo start", transform=ax.transAxes,
                ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _footnote(fig, "cold = calibrate→train (labeled constant, see docs); warm = measured this run")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "startup.png", dpi=DPI)
    plt.close(fig)


def render_all(metrics_path, out_dir):
    doc = json.loads(Path(metrics_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_latency_cdf(doc, out)
    chart_pipeline_breakdown(doc, out)
    chart_headroom(doc, out)
    chart_assets_per_device(doc, out)
    chart_footprint(doc, out)
    chart_startup(doc, out)


if __name__ == "__main__":
    mp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/edge_benchmark/metrics.json")
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else mp.parent
    render_all(mp, od)
    print(f"Rendered charts to {od}")
