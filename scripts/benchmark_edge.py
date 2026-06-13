"""Edge-performance benchmark for EdgeSense inference + the HITL stages.

Measures, on real data through the REAL scoring pipeline (parity-checked against
edgesense.scoring), the per-window latency of each stage (preprocess, forward,
attribution, latent-match), throughput, headroom vs the asset's native rate,
memory footprint, and warm-vs-cold startup. Results append to a per-label entry
in reports/edge_benchmark/metrics.json; charts are regenerated from that JSON.

Environment is a LABEL, not a code path: run the same script under
`taskset -c 0 ... --threads 1 --label dev-1core` for the Pi-class proxy.

  uv run python scripts/benchmark_edge.py --source metropt --label dev-unrestricted
  taskset -c 0 uv run python scripts/benchmark_edge.py --threads 1 --label dev-1core

Pi-class proxy: 1 CPU core, 512 MB — not measured on physical Pi hardware.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROXY_DISCLAIMER = "Pi-class proxy: 1 CPU core, 512 MB — not measured on physical Pi hardware."
# Cold-start (calibrate -> train) wall time for Metro.PT at 5000x, measured in
# the warm-start commit (see docs/EDGE_BENCHMARK.md). Used as a labeled constant;
# pass --include-training to re-measure calibrate->train + one recalibrate here.
STARTUP_COLD_S = 75.0
ALL_SOURCES = ("metropt", "hydraulic", "cmapss")


# ───────────────────────── host / memory helpers ─────────────────────────

def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    return 0.0


def _current_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 2)
    except Exception:
        pass
    return 0.0


def _peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)


def _malloc_trim() -> None:
    """Return freed arena memory to the OS so current RSS reflects live objects
    (glibc retains freed pages after a big transient allocation like a CSV read)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _clean_inference_rss_mb(threads: int, in_features: int = 15) -> float:
    """Measure the deployment-representative inference footprint in a CLEAN
    subprocess: torch + one USAD model + one window, with NO pandas/sklearn or
    dataset loaded. This isolates the real edge process RSS from the benchmark's
    own data machinery (which an edge deployment doesn't run)."""
    import subprocess
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = f'''
import sys, torch
sys.path.insert(0, {src!r})
torch.set_num_threads({threads if threads > 0 else 0} or torch.get_num_threads())
from edgesense.models import USADConv1d, USADConv1dConfig
m = USADConv1d(USADConv1dConfig(in_features={in_features}, base_channels=32, latent_channels=64, downsample_layers=2))
m.eval()
x = torch.zeros(1, 100, {in_features})
with torch.no_grad():
    for _ in range(10):
        r, _, _ = m(x); m.reconstruct_via_decoder2(r); m.encode(x)
rss = 0.0
for line in open("/proc/self/status"):
    if line.startswith("VmRSS:"):
        rss = int(line.split()[1]) / 1024
print(rss)
'''
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        return round(float(out.stdout.strip().splitlines()[-1]), 2)
    except Exception:
        return 0.0


def _host_info(torch) -> dict:
    return {
        "cpu": _cpu_model(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
    }


# ───────────────────────── model load / fresh train ──────────────────────

def _build_scaler(art: dict):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.mean_ = np.asarray(art["scaler"]["mean"], dtype=np.float64)
    sc.scale_ = np.asarray(art["scaler"]["scale"], dtype=np.float64)
    sc.var_ = np.asarray(art["scaler"]["var"], dtype=np.float64)
    sc.n_features_in_ = int(art["scaler"]["n_features_in"])
    return sc


def _build_model(model_config: dict, state_dict):
    from edgesense.models import USADConv1d, USADConv1dConfig
    model = USADConv1d(USADConv1dConfig(**model_config))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def load_or_train(source: str, model_name: str, spec) -> dict:
    """Return {model, scaler, threshold, patterns, origin, calib_samples}.

    Tries the warm artifact first; falls back to a fresh train on real
    calibration data (origin='trained_fresh') so the source still benchmarks.
    """
    from edgesense.sim.artifacts import load_artifact

    try:
        art = load_artifact(source, model_name)
        scaler = _build_scaler(art)
        model = _build_model(art["model_config"], art["model_state"])
        patterns = [
            {"id": p["id"], "centroid": np.asarray(p["centroid"], dtype=np.float32),
             "radius": float(p["radius"])}
            for p in art.get("dismissed_patterns", [])
        ]
        return {
            "model": model, "scaler": scaler, "threshold": float(art["threshold"]),
            "patterns": patterns, "origin": "showcase_artifact",
            "calib_samples": art.get("meta", {}).get("calibration_samples"),
        }
    except FileNotFoundError:
        pass

    # Fresh train on real calibration windows (same windowing as offline).
    from sklearn.preprocessing import StandardScaler
    from edgesense.models import USADConv1d, USADConv1dConfig
    from edgesense.scoring import ScoringConfig, compute_usad_scores
    from edgesense.training import (
        EarlyStoppingConfig, TrainingConfig, seed_all, split_train_validation, train_usad,
    )

    raw = _calibration_windows(source, spec, n=600)
    f = raw.shape[2]
    scaler = StandardScaler().fit(raw.reshape(-1, f))
    scaled = scaler.transform(raw.reshape(-1, f)).reshape(raw.shape).astype(np.float32)
    seed_all(42)
    model = USADConv1d(USADConv1dConfig(in_features=f, base_channels=32, latent_channels=64, downsample_layers=2))
    train_only, val_only = split_train_validation(scaled, val_fraction=0.1)
    train_usad(
        model, train_only,
        TrainingConfig(batch_size=min(256, max(8, train_only.shape[0] // 4)), epochs=25,
                       learning_rate=1e-3, adv_ramp_epochs=15, adv_max_weight=0.3, grad_clip_norm=1.0, seed=42),
        val_windows=val_only,
        early_stopping=EarlyStoppingConfig(patience=6, min_delta=1e-4, max_epochs=25, val_fraction=0.1),
        show_progress=False,
    )
    cal = compute_usad_scores(model, scaled, ScoringConfig(alpha=0.3, beta=0.7), show_progress=False)
    model.eval()
    return {"model": model, "scaler": scaler, "threshold": float(np.percentile(cal, 99.0)),
            "patterns": [], "origin": "trained_fresh", "calib_samples": raw.shape[0]}


# ───────────────────────── real-data windows ─────────────────────────────

def _metropt_features():
    from edgesense.datasets.metropt import load_metropt_dataset
    ds = load_metropt_dataset()
    return ds.data[ds.feature_columns].to_numpy(np.float32), ds.sampling_interval_seconds


def _calibration_windows(source: str, spec, n: int) -> np.ndarray:
    """Real windows from a HEALTHY region, for the fresh-train fallback."""
    return _eval_windows(source, spec, n, region="head")


def _eval_windows(source: str, spec, n: int, region: str = "eval") -> np.ndarray:
    """Build `n` real unscaled windows (n, window_length, F) the same way the
    offline pipeline windows data. `region`='eval' draws from after the
    calibration block; 'head' draws from the start (healthy)."""
    from edgesense.windowing import create_sliding_windows

    wl, stride = spec.window_length, spec.stride
    if source == "metropt":
        feats, _ = _metropt_features()
        start = 0 if region == "head" else min(30000, len(feats) // 3)
        need_rows = wl + (n + 5) * stride
        seg = feats[start:start + need_rows]
        return np.asarray(create_sliding_windows(seg, wl, stride).windows, dtype=np.float32)[:n]

    if source == "hydraulic":
        from edgesense.datasets.hydraulic import load_hydraulic_dataset
        ds = load_hydraulic_dataset()                  # (num_cycles, 60, F) — each cycle is a window
        w = ds.windows
        sel = w[:n] if region == "head" else w[max(0, len(w) - n):]
        return np.asarray(sel, dtype=np.float32)

    if source == "cmapss":
        from edgesense.datasets.cmapss import load_cmapss_fd001
        ds = load_cmapss_fd001()
        units = ds.train_units if region == "head" else ds.test_units
        out: list[np.ndarray] = []
        for uid in sorted(units, key=lambda u: -len(units[u])):
            rows = units[uid][ds.feature_columns].to_numpy(np.float32)
            if rows.shape[0] < wl:
                continue
            out.append(np.asarray(create_sliding_windows(rows, wl, stride).windows, dtype=np.float32))
            if sum(len(o) for o in out) >= n:
                break
        return np.concatenate(out, axis=0)[:n]

    raise ValueError(f"unknown source {source}")


# ───────────────────────── timing core ───────────────────────────────────

def _score_window(model, x, torch, alpha=0.3, beta=0.7):
    """Forward + 0.3/0.7 blend for one (1,T,F) tensor; returns (score, d1, d2)."""
    with torch.no_grad():
        recon1, _, _ = model(x)
        recon2 = model.reconstruct_via_decoder2(recon1)
        d1 = (x - recon1) ** 2
        d2 = (x - recon2) ** 2
        score = float(alpha * d1.mean() + beta * d2.mean())
    return score, d1, d2


def parity_check(model, scaler, raw_windows, torch) -> bool:
    """Assert the harness forward path matches edgesense.scoring within 1e-5."""
    from edgesense.scoring import ScoringConfig, compute_usad_scores

    f = raw_windows.shape[2]
    scaled = np.stack([
        scaler.transform(w).astype(np.float32) for w in raw_windows
    ], axis=0)
    ref = compute_usad_scores(model, scaled, ScoringConfig(alpha=0.3, beta=0.7), show_progress=False)
    mine = np.array([_score_window(model, torch.from_numpy(s[None]), torch)[0] for s in scaled], dtype=np.float32)
    max_abs = float(np.max(np.abs(mine - ref)))
    ok = max_abs <= 1e-5
    print(f"    parity check: max|Δscore| = {max_abs:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _stats(times_s: np.ndarray) -> dict:
    ms = times_s * 1000.0
    return {
        "p50": round(float(np.percentile(ms, 50)), 4),
        "p95": round(float(np.percentile(ms, 95)), 4),
        "p99": round(float(np.percentile(ms, 99)), 4),
        "mean": round(float(np.mean(ms)), 4),
    }


def timed_run(model, scaler, patterns, raw_windows, warmup, n, torch) -> dict:
    perf = time.perf_counter
    pre = np.empty(n); fwd = np.empty(n); attr = np.empty(n); lat = np.empty(n); tot = np.empty(n)
    cents = [p["centroid"] for p in patterns]
    rads = [p["radius"] for p in patterns]

    total = warmup + n
    for i in range(total):
        raw = raw_windows[i % len(raw_windows)]
        timed = i >= warmup
        j = i - warmup

        t0 = perf()
        ts = perf()
        scaled = scaler.transform(raw).astype(np.float32)
        x = torch.from_numpy(scaled[None])
        t_pre = perf() - ts

        ts = perf()
        _, d1, d2 = _score_window(model, x, torch)
        t_fwd = perf() - ts

        ts = perf()
        with torch.no_grad():
            _ = (0.3 * d1.mean(dim=(0, 1)) + 0.7 * d2.mean(dim=(0, 1))).cpu().numpy()
        t_attr = perf() - ts

        ts = perf()
        with torch.no_grad():
            z = model.encode(x).mean(dim=2).squeeze(0).cpu().numpy()
        for c, r in zip(cents, rads):
            if float(np.linalg.norm(z - c)) <= r:
                break
        t_lat = perf() - ts
        t_tot = perf() - t0

        if timed:
            pre[j], fwd[j], attr[j], lat[j], tot[j] = t_pre, t_fwd, t_attr, t_lat, t_tot

    # 0..100th percentile grid of total latency (ms) so the CDF chart is
    # reconstructable from metrics.json without dumping every raw sample.
    cdf = [round(float(v), 4) for v in np.percentile(tot * 1000.0, np.arange(0, 101))]
    return {"preprocess": _stats(pre), "forward": _stats(fwd), "attribution": _stats(attr),
            "latent_match": _stats(lat), "total": _stats(tot), "total_cdf_ms": cdf,
            "_throughput_wps": round(1.0 / float(np.mean(tot)), 2)}


# ───────────────────────── per-source benchmark ──────────────────────────

def benchmark_source(source: str, args, torch) -> dict:
    from edgesense.sim.source import get_source

    src = get_source(source)
    spec = src.spec
    print(f"  [{source}] loading model…")
    t_warm0 = time.perf_counter()
    rss_before = _current_rss_mb()
    bundle = load_or_train(source, args.model, spec)
    rss_after = _current_rss_mb()
    model, scaler, patterns = bundle["model"], bundle["scaler"], bundle["patterns"]
    print(f"    origin={bundle['origin']}  rss_delta={rss_after - rss_before:.1f} MB")

    # Ensure latent_match isn't a no-op: synthesise one pattern if none exist.
    raw = _eval_windows(source, spec, args.windows + args.warmup + 12)
    # Copy the windows into a compact array and free the source dataset so the
    # measured inference footprint reflects edge deployment (model + window
    # buffer + torch), not the benchmark holding the full history in RAM.
    raw = np.array(raw, copy=True)
    gc.collect()
    _malloc_trim()
    if not patterns:
        with torch.no_grad():
            x0 = torch.from_numpy(scaler.transform(raw[0]).astype(np.float32)[None])
            z0 = model.encode(x0).mean(dim=2).squeeze(0).cpu().numpy()
        # radius from spread of a few benign windows so the check is meaningful
        zs = []
        for w in raw[1:9]:
            with torch.no_grad():
                xz = torch.from_numpy(scaler.transform(w).astype(np.float32)[None])
                zs.append(model.encode(xz).mean(dim=2).squeeze(0).cpu().numpy())
        radius = float(np.mean([np.linalg.norm(z - z0) for z in zs])) + 1e-6
        patterns = [{"id": "PAT-synthetic", "centroid": z0.astype(np.float32), "radius": radius}]
        print("    no patterns in artifact -> registered 1 synthetic benign pattern")

    if not parity_check(model, scaler, raw[:10], torch):
        raise SystemExit(f"PARITY CHECK FAILED for {source} — aborting (benchmarking a stripped pipeline)")
    # Warm startup = load model + build first window(s) + first scored window.
    # Only meaningful for the warm (artifact) path; trained_fresh has no artifact.
    warm_load_s = round(time.perf_counter() - t_warm0, 2) if bundle["origin"] == "showcase_artifact" else None

    print(f"    timing {args.windows} windows (after {args.warmup} warmup)…")
    lat = timed_run(model, scaler, patterns, raw, args.warmup, args.windows, torch)
    throughput = lat.pop("_throughput_wps")

    # headroom math — read window/stride/sample-rate from the source, never hardcode.
    cycle_based = spec.cycle_based
    if cycle_based:
        sec_per_cycle = float(getattr(src, "SECONDS_PER_CYCLE", 1)) * float(getattr(spec, "simulated_to_asset_seconds", 1.0))
        required_wps = 1.0 / sec_per_cycle
        rate_note = f"1 window/cycle; real cycle = {sec_per_cycle:.0f}s"
    else:
        _, sampling_s = _metropt_features() if source == "metropt" else (None, 1.0)
        required_wps = (1.0 / sampling_s) / spec.stride
        rate_note = f"sample every {sampling_s:.0f}s, window every {spec.stride} samples"
    headroom = throughput / required_wps

    params = int(sum(p.numel() for p in model.parameters()))
    entry = {
        "model_origin": bundle["origin"],
        "model": {"params": params, "fp32_bytes": params * 4, "rss_delta_mb": round(rss_after - rss_before, 2)},
        "latency_ms": lat,
        "throughput_wps": throughput,
        "required_wps": round(required_wps, 8),
        "headroom": round(headroom, 1),
        "assets_per_core": int(np.floor(headroom)),
        "rate_note": rate_note,
        # peak_rss = ru_maxrss of the whole benchmark process, dominated by
        # pandas loading full datasets to source real windows; NOT the
        # deployment footprint. The edge-representative footprint is the
        # run-level `inference_process_rss_mb` (clean subprocess, no datasets).
        "peak_rss_mb": _peak_rss_mb(),
        "startup_warm_s": warm_load_s,
        "startup_cold_s": STARTUP_COLD_S,
        "training": {"calibrate_train_s": None, "recalibrate_s": None, "peak_rss_mb": None},
        "parity_check": "passed",
        "window_length": spec.window_length,
        "stride": spec.stride,
        "cycle_based": cycle_based,
    }
    print(f"    total p50={lat['total']['p50']}ms p99={lat['total']['p99']}ms | "
          f"{throughput:.0f} win/s vs {required_wps:.2g} required -> {headroom:,.0f}x ({entry['assets_per_core']:,} assets/core)")
    return entry


# ───────────────────────── orchestration ─────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None, help="metropt|hydraulic|cmapss (default: all)")
    ap.add_argument("--model", default="showcase", help="warm artifact name")
    ap.add_argument("--windows", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    ap.add_argument("--label", default="dev-unrestricted")
    ap.add_argument("--out", default="reports/edge_benchmark")
    ap.add_argument("--include-training", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    import torch
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch_baseline_rss_mb = _current_rss_mb()   # RSS of torch runtime before any model/data
    print(f"label={args.label}  threads={torch.get_num_threads()}  windows={args.windows}  "
          f"torch_baseline_rss={torch_baseline_rss_mb:.0f}MB")

    sources = [args.source] if args.source else list(ALL_SOURCES)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        doc = json.loads(metrics_path.read_text())
    else:
        doc = {"schema_version": 1, "runs": {}}

    clean_rss = _clean_inference_rss_mb(args.threads)
    print(f"clean inference-process RSS (torch + model + 1 window, no datasets): {clean_rss:.0f} MB")
    run = {"host": _host_info(torch), "timestamp": datetime.now(timezone.utc).isoformat(),
           "threads": torch.get_num_threads(), "torch_baseline_rss_mb": torch_baseline_rss_mb,
           "inference_process_rss_mb": clean_rss,
           "proxy_disclaimer": PROXY_DISCLAIMER, "sources": {}}

    for source in sources:
        print(f"\n=== {source} ===")
        try:
            entry = benchmark_source(source, args, torch)
        except FileNotFoundError as exc:
            print(f"    SKIP {source}: dataset not available ({exc})")
            continue
        if args.include_training:
            entry["training"] = _measure_training(source, args, torch)
        run["sources"][source] = entry

    doc["runs"][args.label] = run                 # never touches other labels
    metrics_path.write_text(json.dumps(doc, indent=2, default=float))
    print(f"\nWrote {metrics_path}")

    if not args.no_plots:
        try:
            from plot_edge_benchmark import render_all
            render_all(metrics_path, out_dir)
            print(f"Rendered charts to {out_dir}")
        except Exception as exc:
            print(f"(charts skipped: {exc})")


def _measure_training(source: str, args, torch) -> dict:
    """Optional: measure cold calibrate->train + one recalibrate, wall + peak RSS."""
    from sklearn.preprocessing import StandardScaler
    from edgesense.models import USADConv1d, USADConv1dConfig
    from edgesense.scoring import ScoringConfig, compute_usad_scores
    from edgesense.sim.source import get_source
    from edgesense.training import (
        EarlyStoppingConfig, TrainingConfig, seed_all, split_train_validation, train_usad,
    )

    spec = get_source(source).spec
    raw = _calibration_windows(source, spec, n=600)
    f = raw.shape[2]
    t0 = time.perf_counter()
    sc = StandardScaler().fit(raw.reshape(-1, f))
    scaled = sc.transform(raw.reshape(-1, f)).reshape(raw.shape).astype(np.float32)
    seed_all(42)
    model = USADConv1d(USADConv1dConfig(in_features=f, base_channels=32, latent_channels=64, downsample_layers=2))
    tr, va = split_train_validation(scaled, val_fraction=0.1)
    cfg = TrainingConfig(batch_size=min(256, max(8, tr.shape[0] // 4)), epochs=25, learning_rate=1e-3,
                         adv_ramp_epochs=15, adv_max_weight=0.3, grad_clip_norm=1.0, seed=42)
    train_usad(model, tr, cfg, val_windows=va,
               early_stopping=EarlyStoppingConfig(patience=6, min_delta=1e-4, max_epochs=25, val_fraction=0.1),
               show_progress=False)
    calibrate_train_s = time.perf_counter() - t0
    # one recalibrate = retrain on the same windows again
    t1 = time.perf_counter()
    seed_all(42)
    m2 = USADConv1d(USADConv1dConfig(in_features=f, base_channels=32, latent_channels=64, downsample_layers=2))
    train_usad(m2, tr, cfg, val_windows=va,
               early_stopping=EarlyStoppingConfig(patience=6, min_delta=1e-4, max_epochs=25, val_fraction=0.1),
               show_progress=False)
    recalibrate_s = time.perf_counter() - t1
    _ = compute_usad_scores(model, scaled, ScoringConfig(), show_progress=False)
    return {"calibrate_train_s": round(calibrate_train_s, 2), "recalibrate_s": round(recalibrate_s, 2),
            "peak_rss_mb": _peak_rss_mb()}


if __name__ == "__main__":
    main()
