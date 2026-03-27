# GRU-Based Stock Price Predictor (PyTorch)

---

## 🚀 Overview

This project implements a **2-layer GRU (Gated Recurrent Unit)** model in PyTorch to predict the next-day closing price of AAPL stock using multivariate time-series inputs.

The primary goal of this project was to gain a deep, practical understanding of:

- Recurrent Neural Networks (RNNs)
- GRU architecture
- Time-series preprocessing
- GPU-based training workflows
- TensorBoard visualization

This is a clean, end-to-end implementation focused on correct engineering practices rather than financial alpha generation.

---

## 🏗 Model Architecture

Below is the TensorBoard graph visualization of the implemented model:



### Architecture Flow

```
Input (batch, 60, 4)
        ↓
2-Layer GRU (hidden=128, dropout=0.2)
        ↓
Last timestep extraction
        ↓
Linear Layer (128 → 1)
        ↓
Output (Next-day Close)
```

### GRU Configuration

- Input Size: 4 features
- Hidden Size: 128
- Num Layers: 2
- Dropout: 0.2 (between layers)
- Output Size: 1

---

## 📊 Dataset

- **Ticker:** AAPL
- **Date Range:** 2015-01-01 to 2026-03-01
- **Source:** yfinance (`auto_adjust=True`)

### Features Used

- Close (Target)
- High − Low (Daily Volatility Proxy)
- Open − Close (Momentum Proxy)
- Volume / 30-Day MA (Volume Regime Indicator)

---

## 🔄 Data Pipeline

1. Fetch historical data
2. Remove MultiIndex columns (if present)
3. Forward-fill missing values
4. Feature engineering
5. Chronological split (80% train / 10% val / 10% test)
6. MinMaxScaler fitted **only on training data** (no leakage)
7. Create rolling sequences

### Sequence Setup

- Input: 60 past days
- Output: Next-day Close

Final tensor shape:

```
(batch_size, 60, 4)
```

---

## 🧠 Training Configuration

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 50
- Batch Size: 64
- Device: CUDA (RTX 4080)

Training and validation metrics logged using TensorBoard.

Launch TensorBoard:

```
tensorboard --logdir=runs
```

---

## 📈 Evaluation

- Evaluated on unseen test set
- Inverse scaling applied correctly
- Metrics computed:
  - MSE
  - MAE
- Actual vs Predicted plotted using Matplotlib

---

## 🧩 Key Engineering Highlights

- Proper time-series split (no shuffling)
- Zero data leakage during scaling
- Correct tensor dimension management
- Multi-layer GRU implementation
- GPU training workflow
- TensorBoard graph visualization
- Clean modular PyTorch design

---

## ⚠ Limitations

- Predicts raw price instead of returns
- Single-step forecasting only
- No uncertainty estimation
- No walk-forward validation

This project is designed for understanding GRU implementation mechanics rather than building a production trading system.

---

## 🛠 Future Improvements

- Predict log-returns instead of price
- Add rolling volatility features
- Apply gradient clipping
- Add learning rate scheduler
- Multi-horizon forecasting
- Attention mechanism on top of GRU

---

## 📂 Project Structure

```
├── main.ipynb
├── README.md
├── Architecture.png
└── runs/ (TensorBoard logs)
```

---

## 📌 Conclusion

This project demonstrates a structured, leakage-free, GPU-accelerated implementation of a multi-layer GRU model for financial time-series forecasting.

It serves as a solid foundation for mastering RNN-based architectures in PyTorch.

