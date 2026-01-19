<div align="center">

# Informer: A Time Series Forecasting Framework
### High-Efficiency Long-Sequence Forecasting for Financial Markets

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![Paper](https://img.shields.io/badge/AAAI-21-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2012.07436)

<br />

<img src="https://github.com/zhouhaoyi/Informer2020/raw/main/img/informer.png" alt="Informer Architecture" width="80%">

<br />
<br />

**[Abstract](#abstract) • [Methodology](#methodology) • [Installation](#installation) • [Experiments](#experiments) • [Citation](#citation)**

</div>

---

## 📑 Abstract

This repository implements the **Informer** architecture for long-sequence time-series forecasting (LSTF), specifically optimized for **Stock Market Data Analysis**. While traditional Transformer models suffer from high memory consumption and quadratic time complexity, the Informer model leverages a ProbSparse self-attention mechanism to achieve $\mathcal{O}(L \log L)$ complexity.

This project demonstrates the model's capability to capture long-range dependencies in volatile financial datasets, providing accurate predictions for open/close prices and market trends. Furthermore, we address the challenge of non-stationary financial data by integrating distinct encoding techniques that preserve temporal context across extended horizons. The resulting framework not only improves forecast precision but also serves as a scalable backbone for developing automated algorithmic trading strategies.

---

## Methodology

### The Informer Architecture
The core innovation of this project lies in addressing the limitations of the vanilla Transformer when applied to LSTF. We utilize three distinct mechanisms to enhance prediction efficiency:

#### 1. ProbSparse Self-Attention
To handle the quadratic complexity of canonical self-attention, we employ ProbSparse attention. This mechanism selects the "active" queries based on a measurement of Kullback-Leibler divergence, allowing the model to focus only on dominant features.

The standard attention mechanism is defined as:

$$
\mathcal{A}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

In contrast, our **ProbSparse** attention restricts the query set to the top-$u$ dominant queries ($\bar{\mathbf{Q}}$), significantly reducing computational overhead:

$$
\mathcal{A}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\left(\frac{\bar{\mathbf{Q}}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

#### 2. Self-Attention Distilling
To prevent feature redundancy in deep networks, we use a distilling operation that halves the input length in each layer. This operation drastically reduces memory usage while preserving essential information:

$$
\mathbf{X}_{j+1}^t = \mathrm{MaxPool}\left(\mathrm{ELU}\left(\mathrm{Conv1d}\left([\mathbf{X}_j^t]_{\mathrm{AB}}\right)\right)\right)
$$

#### 3. Generative Style Decoder
Unlike standard encoder-decoder structures that generate outputs step-by-step (dynamic decoding), the Informer uses a generative decoder to predict the entire long sequence in a single forward pass. This method effectively mitigates error accumulation during the inference phase of long-sequence forecasting.

---

## 💾 Dataset & Feature Engineering

This project utilizes a high-dimensional financial dataset constructed from **4,452** individual equity instruments. Unlike standard datasets that rely solely on OHLCV data, our feature space is engineered to capture market microstructure, sector rotation, and global macroeconomic correlations.

The model processes a dense input vector of **~100 features per timestamp**, ensuring robust generalization across volatile market regimes.

### 1. Primary Asset Data
For each of the 4,452 stocks, we normalize and ingest the core price action data:
* **OHLCV:** Open, High, Low, Close, Volume.
* **Log Returns:** $\ln(P_t / P_{t-1})$ for stationarity.

### 2. Technical Indicators (Momentum & Volatility)
We compute proprietary technical signals to feed the attention mechanism with trend-aware context:
* **Trend:** Exponential Moving Averages (EMA 9, 21, 50, 100), MACD (Line, Signal, Hist).
* **Momentum:** RSI (14), Efficiency Ratio.
* **Volatility:** Average True Range (ATR), Rolling Volatility (10, 20-day windows), Sharpe Ratio (20-day).
* **Statistical Features:** Z-Scores for Volume and Returns to normalize outliers.

### 3. Macro-Economic Context (Global & Local)
To solve the "isolated asset" bias, we inject broad market health indicators directly into the feature vector:
* **Indian Benchmarks:** NIFTY 50, SENSEX, FinNifty.
* **Sectoral Indices:** NIFTY IT, Pharma, Bank, Auto, Metal, Energy, Realty, PSU Bank.
* **Global Correlations:** S&P 500 (USA), NASDAQ, Dow Jones, FTSE (UK), DAX (Germany), CAC40 (France), Nikkei (Japan), Hang Seng, Shanghai, Taiwan Weighted.

### 4. Lagged Temporal Features
To capture immediate past dependencies explicitly before the Transformer layers:
* **Price Lags:** $t_{-1}, t_{-2}, t_{-3}, t_{-4}$ percentage changes.
* **Volume Lags:** Volume changes over the last 4 distinct time steps.

> **Data Scale:** The final dataset comprises millions of data points, processed with Zero-Mean Unit-Variance normalization to stabilize the **ProbSparse** attention mechanism.

## ⚙️ Model Architecture & Configuration

The model is configured to handle high-dimensional inputs with a specific focus on short-term precision (`out_len=1`) using a full-attention mechanism. Below is the detailed hyperparameter configuration used for the final training runs.

### 🔌 Input/Output Tensor Structure
The Informer model processes four specific tensors during the forward pass:

* **`x_enc` (Encoder Input):** The historical sequence of **100 features** (Open, Close, Indicators, Macro Indices).
    * *Shape:* `(Batch, 96, 100)`
* **`x_mark_enc` (Encoder Time Features):** Time-stamps encoded as embeddings (Day, Month, Weekday).
    * *Shape:* `(Batch, 96, Features)`
* **`x_dec` (Decoder Input):** A concatenation of the "Label" (start token) and zero-padding for the target.
    * *Shape:* `(Batch, 48 + 1, 1)`
* **`x_mark_dec` (Decoder Time Features):** Future time-stamps for the prediction horizon.
    * *Shape:* `(Batch, 48 + 1, Features)`

### 🛠 Hyperparameter Specification

| Category | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Data Dimensions** | `enc_in` | **100** | Input feature dimension (Stock + Indicators + Macro) |
| | `dec_in` / `c_out` | **1** | Output dimension (Predicting `% Change`) |
| | `seq_len` | **96** | Input sequence length (Lookback window) |
| | `label_len` | **48** | Start token length for Generative Decoder |
| | `pred_len` | **1** | Prediction horizon (Next-day forecast) |
| **Architecture** | `d_model` | **128** | Dimension of the model embeddings |
| | `n_heads` | **4** | Number of Multi-Head Attention heads |
| | `e_layers` | **2** | Number of Encoder layers |
| | `d_layers` | **1** | Number of Decoder layers |
| | `d_ff` | **512** | Dimension of Fully Connected layer |
| **Mechanism** | `attn` | **'full'** | Attention mechanism (Full Attention used for precision) |
| | `embed` | **'timeF'** | Time-feature encoding strategy |
| | `activation` | **'gelu'** | Activation function |
| | `distil` | **False** | Distilling operation (Disabled for short sequences) |
| **Training** | `batch_size` | **64** | Batch size for gradient descent |
| | `dropout` | **0.1** | Dropout rate for regularization |
| | `device` | **CUDA** | GPU Acceleration |

> **Configuration Note:** Unlike the standard Informer which uses `prob` attention for extreme long sequences, this configuration utilizes **`attn='full'`**. Since our prediction horizon is short (`pred_len=1`), Full Attention provides superior granularity and accuracy compared to sparse approximations, while the Informer's Generative Decoder structure prevents error accumulation.


## Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed.

```bash
git clone [https://github.com/yourusername/informer-stock-forecasting.git](https://github.com/yourusername/informer-stock-forecasting.git)
cd informer-stock-forecasting
pip install -r requirements.txt
