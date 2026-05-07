# Low-Resource Blocking for Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for **"Low-Resource Adaptation Pipeline for Entity Resolution using Ultra-Lightweight Bi-Encoders"**.

This repository contains the code, evaluation scripts, and data utilities to reproduce a single-stage, low-resource adaptation pipeline utilizing Ultra-Lightweight Bi-Encoders (ULBs) to achieve high-fidelity entity blocking with strictly limited annotation budgets.

## 📖 Overview

Entity Resolution (ER) is a vital process in modern data integration, linking disparate records that correspond to the same real-world entity. While large pre-trained language models have advanced ER, state-of-the-art frameworks increasingly rely on heavyweight, two-stage pipelines featuring heavy Transformer architectures and computationally expensive cross-encoder matching modules. These designs introduce severe operational bottlenecks, making them prohibitively slow for enterprise-scale deployment. In this work, we propose a low-resource adaptation pipeline utilizing Ultra-Lightweight Bi-Encoders (ULBs) to achieve high-fidelity entity blocking with strictly limited annotation budgets. By employing schema-agnostic serialization and online contrastive loss with in-batch hard-negative mining, our pipeline rapidly warps the dense vector space to penalize false positives. Evaluated across multiple benchmarks using only a 30\% training budget, our ULBs maintain highly competitive top-$k$ recall while operating with roughly 80\% fewer parameters than leading baselines. 

## ✨ Key Features

* **Ultra-Lightweight Architecture:** Operates with over **75% fewer parameters** than leading `roberta-base` baselines (e.g., Sudowoodo, CLER).
* **Data Efficiency:** Achieves state-of-the-art Top-k recall using highly constrained data budgets (e.g., 30% of the training space).
* **Single-Stage Sub-linear Retrieval:** Executes retrieval directly on dense spaces without heavy cross-attention mechanisms, rendering the secondary matching stage entirely obsolete.

## 🚀 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/dimkar121/Low-Resource-Blocking.git](https://github.com/dimkar121/Low-Resource-Blocking.git)
cd Low-Resource-Blocking
pip install -r requirements.txt
