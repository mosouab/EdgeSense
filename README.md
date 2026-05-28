# EdgeSense

## Introduction
Unsupervised anomaly detection for predictive maintenance on industrial assets, trained and evaluated on the Metro.PT air-compressor dataset (Veloso et al., 2022). Built as a proof of concept for an entrepreneurship project at uni.

## Problem
Industrial compressors fail expensively. Cloud-based monitoring pipelines stream raw multivariate sensor data continuously, paying for bandwidth, latency and data-sovereignty risk. Supervised approaches also need labeled failure data, which is rare and expensive on real assets. The goal here is a model that learns an asset's healthy operating envelope from unlabeled sensor windows, runs at the edge, and flags deviations as they emerge.

## Architecture
USAD-style network (Audibert et al., 2020) with a 1D-CNN backbone: a shared encoder feeds two decoders. Training has two coupled phases: a reconstruction phase, then a minimax adversarial game where decoder 2 learns to be bad at reconstructing decoder 1's output. At inference, the disagreement between the two decoders amplifies the anomaly score on out-of-distribution windows.

![Architecture](figures/architecture.png)

`w_adv` ramps linearly from 0 to 0.3 over 30 epochs, with gradient clipping at norm 1.0 and best-checkpoint restoration. The threshold is set after deployment from the 99th percentile of scores on a 14-day on-site recalibration window (no failure labels are used). All evaluation runs on a held-out horizon.

## Results

Test horizon 2020-04-15 to 2020-09-01, six failures (four original Metro.PT, two audit-confirmed).

| | Recall | Precision | F1 | AUC |
|---|---|---|---|---|
| Recalibrated (deployable) | 0.883 | 0.572 | 0.69 | 0.905 |
| Oracle PR-optimal | 0.955 | 0.519 | 0.67 | 0.905 |
| Training-period (no recalibration) | 0.906 | 0.236 | 0.37 | 0.905 |

Anomaly score across the test horizon, with the recalibration window (green), labeled failures (red) and both thresholds:
![Anomaly score timeline](figures/03_anomaly_score_timeline.png)

Score distribution, healthy vs failure:
![Score distribution](figures/04_score_distribution.png)

Precision-recall curve:
![PR curve](figures/05_precision_recall_curve.png)

Latent space (PCA of encoder output):
![Latent PCA](figures/06_latent_pca.png)

Training dynamics:
![Training curves](figures/02_training_curves.png)

June air-leak event, detection latency 6 min 15 s after the labeled start:
![June zoom](figures/07_june_failure_zoom.png)

Sensor traces, healthy day vs failure day:
![Sensor overview](figures/01_sensor_overview.png)

Audit of the top three unlabeled high-score plateaus (used to confirm the two added failures):
![Plateau audit](figures/08_unlabeled_plateau_audit.png)

## Literature
- Audibert, J., Michiardi, P., Guyard, F., Marti, S., and Zuluaga, M.A. (2020). USAD: UnSupervised Anomaly Detection on Multivariate Time Series. *KDD 2020*.
- Veloso, B., Ribeiro, R.P., Gama, J., and Pereira, P. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data*, 9, 764.
- Kim, S., Choi, K., Choi, H.S., Lee, B., and Yoon, S. (2022). Towards a rigorous evaluation of time-series anomaly detection. *AAAI 2022*.
- Fawaz, H.I., Forestier, G., Weber, J., Idoumghar, L., and Muller, P.A. (2019). Deep learning for time series classification: a review. *Data Mining and Knowledge Discovery*, 33, 917-963.
