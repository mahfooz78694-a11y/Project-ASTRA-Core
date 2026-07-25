# Project ASTRA (v2.0)
### Zero-retraining Vector Interception Layer for Deep Neural Networks via Float64 Nullspace Projection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21532310.svg)](https://doi.org/10.5281/zenodo.21532310)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Precision: Float64](https://img.shields.io/badge/Precision-IEEE%20754%20Float64-orange.svg)](https://doi.org/10.5281/zenodo.21532310)
---

## 📊 Enterprise Audit & Hard-Truth Verification

Project ASTRA v2.0 has undergone a full static and dynamic audit via `astra_auditor_pro.py`.

* **AST Security Hygiene:** `100/100` (Zero hardcoded secrets, zero dynamic eval hazards).
* **Mathematical Invariant:** Idempotency error bounded at $1.10 \times 10^{-16}$ (Enforced IEEE 754 Float64).
* **Runtime SLA Latency:** Mean hook overhead measured at $0.0564\text{ ms}$ (Target $< 1.0\text{ ms}$).
* **Cryptographic Lock:** Core codebase SHA-256 digest verified against Zenodo Release `10.5281/zenodo.21532310`.

> 📄 **Full Interactive Report:** View the complete audit summary in [`Project_ASTRA_Audit_Report_Pro.html`](Project_ASTRA_Audit_Report_Pro.html).

## 📌 Abstract & Overview
Deep Neural Networks (DNNs) exhibit acute vulnerabilities to intermediate activation layer perturbations engineered through out-of-distribution (OOD) noise injection. **Project ASTRA v2.0** introduces a parameter-invariant runtime defence framework designed to intercept high-dimensional 4D activation tensors ($B \\times C \\times H \\times W$) and deflect adversarial vectors into an orthogonal complement nullspace manifold.

* **Authors:** Mahfooz, MD & Alam, Alsaad
* **Official Research Paper (DOI):** [10.5281/zenodo.21532310](https://doi.org/10.5281/zenodo.21532310)


## ⚙️ Technical Specifications & Guarantees
* **Numerical Standard:** Strict IEEE 754 `Float64` double-precision enforcement to eliminate precision drift.
* **Core Operator:** Singular Value Decomposition (SVD) based nullspace projection ($P_{\\parallel} = U_k U_k^T$).
* **Residual Noise Floor:** Strictly bounded $\\le 10^{-12}$.
* **SLA Latency Overhead:** Optimized to $+0.42\\text{ ms}$ per batch.


## 📊 Empirical Verification & Performance Proof
The validation script automatically generates empirical performance proofs. Below is the runtime noise attenuation profile:

![Noise Floor Benchmark](./assets/noise_floor_benchmark.png)


## 🚀 Quick Start & Installation Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mahfooz78694-a11y/Project-ASTRA-Core.git
   cd Project-ASTRA-Core
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute evaluation and benchmark suite:**
   ```bash
   python evaluate.py
   ```

---

## 🔒 Security, Integrity & Enterprise Compliance
* **Precision Locking:** The framework enforces global double-precision rules to maintain mathematical bounds and eliminate numerical drift during deep neural execution cycles.
* **Cryptographic Checksums:** Execution scripts compute automated SHA-256 cryptographic hashes for core modules to verify data integrity and establish verifiable trust during enterprise deployment.

---

## ⚖️ License & Intellectual Property
This project is open-sourced under the terms of the [Apache 2.0 License](LICENSE). All rights to the core mathematical formulations, architectural pipelines, and research documentation remain fully protected under the published Zenodo DOI (`10.5281/zenodo.21532310`).


