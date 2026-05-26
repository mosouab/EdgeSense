# Literature Review & Theoretical Foundation

This document synthesizes the key academic papers and industry reports that form the theoretical and architectural foundation of the **EdgeSense** project. The selected literature spans predictive maintenance frameworks, deep learning for time series, adversarial autoencoders, and the transition toward Edge AI/TinyML.

## 1. The Industrial Context: Predictive Maintenance in Industry 4.0
**Source:** *Predictive Maintenance in Industry 4.0* (Sang et al., 2020)

This paper establishes the macro-architectural requirements for modern predictive maintenance (PdM). In the context of Industry 4.0, manufacturing has shifted from isolated processes to highly collaborative, data-driven ecosystems.
* **The Evolution of Maintenance:** The industry is moving from reactive (run-to-failure) and scheduled (preventive) maintenance to condition-based and **predictive maintenance**, utilizing machine learning to detect pending failures before they occur.
* **Data Challenges:** Industrial environments generate massive, heterogeneous datasets (vibration, temperature, logs) that require modular and scalable architectures (like RAMI 4.0 and FIWARE) to process effectively.
* **Security & Decentralization:** The paper explores secure data exchange using Industrial Data Spaces (IDS) and Blockchain to ensure data sovereignty, transparency, and trust across collaborative supply chains.

## 2. The Algorithmic Shift: Deep Learning for Time Series
**Source:** *Deep learning for time series classification: a review* (Fawaz et al., 2019)

Traditional statistical methods often fail to capture the complex, non-linear dependencies of multivariate sensor data. This comprehensive review evaluates various Deep Neural Network (DNN) architectures for time-series analysis.
* **Why Convolutional Neural Networks (CNNs)?** The study validates that 1D-CNNs and Residual Networks (ResNets) frequently outperform recurrent networks (like LSTMs) in time-series tasks. CNNs effectively learn spatially invariant temporal features without suffering from the vanishing gradient problems and high computational costs associated with LSTMs.
* **Efficiency:** CNNs parallelize well, offering significantly faster training and inference times, which is a critical prerequisite for edge computing.
* **Interpretability:** The paper highlights the use of Class Activation Maps (CAM) to reduce the "black-box" nature of DNNs, allowing engineers to visualize which specific temporal segments triggered a classification or anomaly alert.

## 3. The Core Engine: UnSupervised Anomaly Detection (USAD)
**Source:** *USAD: UnSupervised Anomaly Detection on Multivariate Time Series* (Audibert et al., 2020)

This paper introduces the exact mathematical and architectural framework implemented in EdgeSense (`USADConv1d`). Detecting anomalies in industrial data is challenging because failures are rare, making supervised learning impossible.
* **The Over-Generalization Problem:** Standard Autoencoders (AEs) often generalize too well, successfully reconstructing anomalous inputs and thereby failing to detect them (false negatives).
* **The USAD Architecture:** USAD solves this by combining autoencoders with Generative Adversarial Networks (GANs) principles. It uses a shared encoder and two decoders trained in a two-phase loop:
  1. **Reconstruction Phase:** Both decoders learn to reconstruct normal data.
  2. **Adversarial Phase:** Decoder 1 tries to fool Decoder 2, while Decoder 2 tries to differentiate between real data and data reconstructed by Decoder 1.
* **The Result:** This adversarial minimax game explicitly amplifies the reconstruction error for abnormal data. USAD offers the stability of autoencoders (avoiding GAN mode collapse) while delivering state-of-the-art anomaly detection speed and accuracy.

## 4. The Deployment Reality: The TinyML Imperative
**Source:** *Unsupervised Anomaly Detection in Predictive Maintenance: General Architectures, Edge Deployment, and State-of-the-Art Approaches*

This comprehensive report explores the mathematical complexities of industrial data (non-stationarity, extreme class imbalance) and the trajectory of anomaly detection models from classical baselines (Isolation Forests) to heavy Transformers (TranAD, Anomaly Transformer).
* **The Cloud Bottleneck:** Transmitting high-frequency, multivariate vibration data to the cloud incurs exorbitant bandwidth costs, introduces latency, and violates strict corporate data privacy policies.
* **The Edge/TinyML Solution:** The report strongly argues for decentralizing AI inference directly onto resource-constrained microcontrollers attached to the physical sensors (Edge AI). 
* **Synthesis:** By deploying highly compressed, unsupervised anomaly detection models (like 1D-CNN Autoencoders) locally, systems can achieve autonomous, ultra-low latency monitoring without transmitting raw telemetry off-site.

---
### 🔗 Relevance to EdgeSense
The EdgeSense architecture is a direct materialization of these findings:
1. It avoids heavy, cloud-dependent LSTMs/Transformers in favor of efficient **1D-CNNs**.
2. It operates entirely **unsupervised** using the **USAD adversarial training** protocol to reliably amplify anomaly signals.
3. It focuses on lightweight, sliding-window inference optimized for **localized Edge AI** deployment, bypassing bandwidth and latency bottlenecks.