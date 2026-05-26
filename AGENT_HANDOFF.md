# EdgeSense Handoff Notes

This document summarizes the work completed, decisions made, and how to continue the project.

## Scope Completed

### 1. Data Ingestion
- **File:** `src/edgesense/data_ingestion.py`
- **Key decisions:**
  - Load `data/MetroPT3(AirCompressor).csv`, drop unnamed index columns, enforce numeric types.
  - Use `timestamp` as the time axis; sort and validate monotonicity.
  - Exposed `MetroPTDataset` dataclass with metadata (sampling interval, start/end time).
  - Added failure intervals from the official dataset PDF into `load_failure_reports()`.
- **Usage:** `load_metropt_dataset()` and `load_failure_reports()`.

### 2. Preprocessing Pipeline (Scaling + Missing Values)
- **File:** `src/edgesense/preprocessing.py`
- **Key decisions:**
  - Use **StandardScaler** (consistent with later linear output in the AE) and fit **only on healthy data** (failure intervals excluded).
  - Missing values are interpolated linearly and forward/backward filled.
  - `MetroPTPreprocessor` supports `fit`, `transform`, `fit_transform`, and save/load via pickle.
  - `build_healthy_mask()` creates a per-row mask from failure intervals.

### 3. Sliding Window Utilities
- **File:** `src/edgesense/windowing.py`
- **Key decisions:**
  - Windows are `(num_windows, window_size, num_features)` to match PyTorch `(B, T, F)`.
  - `build_window_mask()` aggregates a per-row healthy mask into a per-window mask.
  - `create_sliding_windows()` optionally returns window start/end timestamps.
  - `iter_sliding_windows()` yields windows lazily for memory efficiency.

### 4. 1D-CNN USAD Model
- **Files:** `src/edgesense/models/usad_cnn.py`, `src/edgesense/models/__init__.py`
- **Key decisions:**
  - USAD architecture: **shared encoder + two decoders** (USAD-style).
  - Convolutional encoder + upsample-convolution decoders for edge-friendly inference.
  - Input/Output shape: `(batch, seq_len, features)`; internal conv uses `(B, F, T)`.
  - No output activation (data is StandardScaler-normalized).
  - Decoder uses nearest-neighbor upsampling + conv; final interpolation ensures exact length.
  - Shape checks and parameter validation are enforced.

### 5. README Updates
- **File:** `README.md` (renamed from `readme.md`)
- **Key decisions:**
  - Added a TODO checklist with current progress.
  - Normalized case to avoid cross-platform README conflicts.

### 6. Git / LFS / Repo Hygiene
- **Dataset:** `data/MetroPT3(AirCompressor).csv` is tracked via **Git LFS** to bypass GitHub size limits.
- **Docs:** PDFs in `docs/` are committed normally.
- **.gitignore:** added `.python-version` and `__pycache__/`.

## Project Structure Overview

```
src/edgesense/
  __init__.py
  data_ingestion.py
  preprocessing.py
  windowing.py
  models/
    __init__.py
    usad_cnn.py
```

## How to Run the Current Pipeline

```bash
uv run python main.py
```

This prints:
- dataset size, time range, median sampling interval
- preprocessing summary (scaled shape + healthy/unhealthy counts)

## Dependencies

Managed with **uv**. `torch` was added via:
```bash
uv add torch
```

This updates `pyproject.toml` and `uv.lock`.

## Important Decisions & Rationale

1. **StandardScaler vs MinMax:**
   - StandardScaler keeps data centered, so the decoder output remains linear (no sigmoid).
2. **Healthy-only fitting:**
   - Avoids leakage of failure dynamics into the scaler statistics.
3. **Convolutional USAD:**
   - Better edge efficiency vs LSTM while retaining window-based anomaly scoring.
4. **Dual decoders:**
   - Mirrors USAD’s adversarial training setup.
5. **Git LFS for CSV:**
   - Required for GitHub’s 100 MB file limit.


## Next Steps (Recommended)

1. **Training Loop (`training-loop` todo):**
   - Implement USAD’s two-phase loss:
     - Phase 1: both decoders minimize reconstruction error.
     - Phase 2: AE1 minimizes and AE2 maximizes reconstruction on AE1 output.
2. **Anomaly Scoring:**
   - Weighted score: `alpha * MSE(x, AE1(x)) + beta * MSE(x, AE2(AE1(x)))`.
   - Threshold from healthy calibration windows (e.g., percentile).
3. **Evaluation:**
   - Use failure intervals from `load_failure_reports()` to validate detection.

## Known Constraints

- The model assumes `sequence_length >= 2 ** downsample_layers`.
- LFS must be installed for fresh clones to pull the CSV data.

