# EdgeSense Handoff Notes

This document summarizes the current state of the EdgeSense project, recent achievements, decisions made, and pending tasks for future development sessions.

## 🚀 Current Project State: Highly Advanced POC
The core machine learning pipeline for EdgeSense is complete, highly optimized, and competition-ready. The project successfully implements an edge-native, unsupervised anomaly detection system using a 1D-CNN USAD (Unsupervised Anomaly Detection via Adversarial Training) architecture.

### Key Achievements
* **1D-CNN USAD Architecture:** Successfully implemented the dual-decoder adversarial training framework described by Audibert et al. (2020), optimized with 1D-CNNs for time-series data to reduce memory footprint and improve inference speed on edge devices.
* **Exceptional Performance Benchmarks:** Validated on the Metro.PT dataset, the model achieved:
  * **Recall: 100.0%** (Comprehensive detection of all critical failure events).
  * **Precision (Point-Adjusted): 59.1%** (Exceeded the >50% target).
  * **ROC-AUC: 0.9795** (Excellent separation between healthy states and failures).
* **Optimization Strategies Applied:**
  * **Macro-Context Windowing:** Increased window size to 100s with a 50s stride to help the model learn the macro-physics of the machine cycles rather than transient micro-spikes.
  * **High-Adversarial Scoring:** Shifted scoring weights to $\alpha=0.3, \beta=0.7$, heavily relying on the adversarial reconstruction discrepancy ($AE_2(AE_1(x))$) to amplify true anomalies and suppress normal operational variance.
  * **Post-Processing Pipeline:** Implemented a robust filtering system consisting of a Median Filter (window=11) for score smoothing and Temporal Persistence (min_consecutive=25) to effectively eliminate false positive noise bursts.
* **Documentation & Visualization:**
  * The `README.md` has been rewritten to be high-level, competition-ready, and focuses on the strategic value of edge deployment.
  * Added `docs/LITERATURE_SUMMARY.md` to ground the project in academic research.
  * Generated compelling visualizations, notably a `binary_fault_timeline.png` showing perfect recall alignment and a `june_failure_zoom.png` providing granular proof of low-latency detection.

## 🛑 What Was Started But Abandoned (The Simulation/UI Layer)
In the latter part of the session, an attempt was made to build a "Live Simulation" layer to demonstrate real-time edge inference and visualize it through a modern web UI. **This was aborted by the user and the files were deleted.**

**The attempted architecture (for future reference):**
1. **MQTT Broker:** HiveMQ (public) used for decoupled communication.
2. **Backend Engine (`simulation/engine.py`):** An asynchronous Python script designed to stream the CSV data, autonomously calibrate the model on the first 2000 samples, calculate its own threshold, and then seamlessly shift into monitoring mode, broadcasting telemetry and anomaly scores via MQTT.
3. **Frontend HMI (`frontend/src/App.tsx`):** A React/TypeScript/Vite app with TailwindCSS and Apache ECharts designed to subscribe to the MQTT broker and visualize the live data, including a "Time Machine" scrubber to jump to failure events.

*Reason for abandonment:* Issues with ECharts rendering dimensions within the Vite/React setup and general complexity overhead. The user opted to remove the simulation files and focus on the completed ML pipeline.

## 📋 Next Steps / How to Continue

1. **Re-attempt UI / Simulation (Optional):** If the requirement arises to demonstrate the model live, the architecture outlined above (decoupled Python MQTT publisher + separate UI subscriber) is the correct path. Future attempts should focus on ensuring ECharts instances have explicit container dimensions upon initialization.
2. **Edge Deployment Packaging:** The logical next step for the core ML pipeline is to package the `usad_conv1d.pt` model and the `GenericPreprocessor` artifacts for deployment on actual edge hardware (e.g., converting to ONNX or TensorRT, writing a lightweight C++ or Rust inference wrapper if Python is too heavy for the target IPC).
3. **Multi-Machine Validation:** While the pipeline is now structure to be "Universal" (dynamic thresholding during a calibration phase), it should be validated against a completely different industrial dataset to prove its machine-agnostic capabilities.

## 📂 Key Files to Know
* `src/edgesense/models/usad_cnn.py`: The core 1D-CNN dual-decoder architecture.
* `src/edgesense/training.py`: The adversarial training loop with evolving loss weights.
* `src/edgesense/scoring.py`: Contains the critical $\alpha=0.3, \beta=0.7$ scoring logic.
* `src/edgesense/evaluation.py`: Houses the essential Median Smoothing and Temporal Persistence filters that made >50% precision possible.
* `scripts/run_full_evaluation.py`: The master script to train, evaluate, and generate all report artifacts.
* `docs/LITERATURE_SUMMARY.md`: The theoretical backing of the project.