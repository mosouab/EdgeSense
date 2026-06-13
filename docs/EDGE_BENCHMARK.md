# Edge-performance benchmark

Demonstrated, reproducible evidence that EdgeSense inference (and the HITL
stages) fits on cheap edge hardware. Pitch bar, not research bar — every number
is labeled with the environment it was measured in.

**Pi-class proxy: 1 CPU core, 512 MB — not measured on physical Pi hardware.**

## What it measures

Per scored window, through the **real** scoring pipeline (parity-checked against
`edgesense.scoring` within 1e-5 — not a stripped path), on **real eval-region
windows** built with the offline windowing:

- **preprocess** — scaler transform + tensor conversion
- **forward** — USAD forward + the 0.3·AE1 + 0.7·AE2 score
- **attribution** — per-feature contribution
- **latent_match** — encoder latent + dismissed-pattern distance (Layer 3)

plus throughput vs the asset's **native rate** (read from the source spec, never
hardcoded), memory footprint, and warm-vs-cold startup.

## Reproduce

`models/showcase/metropt__showcase.pt` is git-ignored; regenerate it first (or
Metro.PT falls back to a fresh train, labeled `trained_fresh`):

```
uv run python scripts/run_simulation.py          # terminal 1 (for the showcase model)
uv run python scripts/pretrain_showcase.py        # terminal 2 — writes the Metro.PT warm model

# reference (dev machine, all cores):
uv run python scripts/benchmark_edge.py --label dev-unrestricted

# headline Pi-class proxy (1 core, 1 torch thread):
taskset -c 0 uv run python scripts/benchmark_edge.py --threads 1 --label dev-1core

# charts only (no re-benchmark — reads metrics.json):
uv run python scripts/plot_edge_benchmark.py
```

Results append per-label to `reports/edge_benchmark/metrics.json` (a run never
overwrites another label's entry). Charts → `reports/edge_benchmark/*.png`.

## Host (this capture)

`13th Gen Intel Core i7-1355U`, 15.3 GB RAM, Linux 6.17, PyTorch 2.12.0.
Hydraulic and CMAPSS had no showcase artifact in this run, so they were
`trained_fresh` (same architecture → same latency; provenance is labeled).

## Captured run — `dev-1core` (taskset -c 0, torch threads = 1)

```
label=dev-1core  threads=1  windows=2000  torch_baseline_rss=458MB
clean inference-process RSS (torch + model + 1 window, no datasets): 558 MB

=== metropt ===
  [metropt] loading model…
    origin=showcase_artifact  rss_delta=5.8 MB
    parity check: max|Δscore| = 1.49e-08  -> PASS
    timing 2000 windows (after 100 warmup)…
    total p50=0.5582ms p99=0.9872ms | 1681 win/s vs 0.002 required -> 840,630x (840,630 assets/core)

=== hydraulic ===
  [hydraulic] loading model…
    origin=trained_fresh  rss_delta=141.6 MB
    no patterns in artifact -> registered 1 synthetic benign pattern
    parity check: max|Δscore| = 2.98e-08  -> PASS
    timing 2000 windows (after 100 warmup)…
    total p50=0.4812ms p99=0.7988ms | 1964 win/s vs 0.017 required -> 117,829x (117,829 assets/core)

=== cmapss ===
  [cmapss] loading model…
    origin=trained_fresh  rss_delta=35.9 MB
    no patterns in artifact -> registered 1 synthetic benign pattern
    parity check: max|Δscore| = 1.19e-07  -> PASS
    timing 2000 windows (after 100 warmup)…
    total p50=0.4417ms p99=0.9129ms | 2032 win/s vs 4.6e-05 required -> 43,889,472x (43,889,472 assets/core)

Wrote reports/edge_benchmark/metrics.json
Rendered charts to reports/edge_benchmark
```

## Headline numbers (dev-1core, from metrics.json)

| Source | p50 | p99 | throughput | required by asset | headroom | assets / core |
|---|---|---|---|---|---|---|
| Metro.PT compressor | 0.558 ms | 0.987 ms | 1681 win/s | 0.002 win/s | 840,630× | 840,630 |
| Hydraulic rig | 0.481 ms | 0.799 ms | 1964 win/s | 0.0167 win/s | 117,829× | 117,829 |
| CMAPSS turbofan | 0.442 ms | 0.913 ms | 2032 win/s | 0.0000463 win/s | 43,889,472× | 43,889,472 |

Model: ~41.5k parameters, **0.166 MB** fp32. Warm start **2.13 s** vs ~75 s cold
(~35×). Stage means (Metro.PT): preprocess 0.087 ms, forward 0.384 ms,
attribution 0.024 ms, latent-match 0.099 ms — the forward pass dominates; the
two HITL stages add ~0.12 ms combined.

> **On single-core being faster than 10 threads.** dev-1core p50 (0.56 ms) beats
> dev-unrestricted (1.38 ms): these ops are sub-millisecond, so OpenMP thread
> dispatch costs more than it saves. One core is the right deployment shape here.

## The headroom is real — here's the math

The numbers look absurd because the assets are slow. Metro.PT samples once every
10 s and we score a new window every 50 samples → a window is *required* every
**500 s** (0.002 windows/s). One core delivers 1681 windows/s:

```
headroom = 1681 / (1/500) = 1681 × 500 = 840,630×
```

CMAPSS is per-cycle and a real flight cycle is ~6 h (21,600 s), so one window is
required every 21,600 s (4.63e-5 win/s); 2032 win/s ÷ 4.63e-5 ≈ **43.9 M×**.
These are not GPU numbers or batched throughput — single process, one window at a
time, full pipeline including attribution + Layer-3 latent match. Verified twice;
kept as measured.

## Footprint — the one honest caveat

The **model is 0.166 MB** (41.5k fp32 params) — it imposes no memory pressure.

But the fp32 **PyTorch x86 inference process is ~558 MB**, *over* the 512 MB
budget. That is **not** EdgeSense: PyTorch's x86 build maps ~458 MB of
MKL/libtorch runtime into RSS before any model exists (`torch_baseline_rss_mb =
458`), and `smaps_rollup` confirms ~550 MB is private (not shared library pages).

This is the single number where the x86 proxy is **not** Pi-representative — an
ARM PyTorch build or, better, an ONNX Runtime / TFLite deployment carries a far
smaller runtime. Quantization/ONNX is an explicit non-goal of this pass (fp32
numbers carry the latency story), so memory-fit is **deferred** to a runtime-
packaging pass. We report it over budget rather than hide it.

`peak_rss_mb` in metrics.json (~1.2 GB) is higher still and is **not** a
deployment number — it's the benchmark process holding full historical datasets
in RAM to source real windows; an edge device streams windows and never loads
the dataset. The deployment-representative figure is `inference_process_rss_mb`
(clean subprocess: torch + model + one window).

## Charts (`reports/edge_benchmark/`)

1. `latency_cdf.png` — per-window latency CDF, one line per environment, one panel per source.
2. `pipeline_breakdown.png` — stacked stage means (incl. attribution + latent match).
3. `headroom.png` — achieved vs required windows/s (log), with the real-time ratio.
4. `assets_per_device.png` — assets one 1-core device can host (the pitch slide).
5. `footprint.png` — model fp32 size + inference-process RSS vs the 512 MB line.
6. `startup.png` — cold (~75 s) vs warm (2.1 s).
