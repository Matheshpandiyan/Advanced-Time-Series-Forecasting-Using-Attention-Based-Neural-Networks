# Attention Time-Series Project (Fixed)

This repository contains a corrected and reproducible implementation for time-series forecasting using:
- Baseline LSTM model
- Transformer-like TimeSeriesTransformer with attention capture
- Rolling-origin cross-validation (time-series CV)
- Metrics logging (MAE, RMSE, MAPE)
- Early stopping, scheduler, checkpointing
- Attention extraction and plotting utilities
- Report template (REPORT.md)

## Quick start (assumes Python 3.8+, PyTorch installed)
1. (Optional) Create a virtualenv and install PyTorch.
2. Generate demo data (if you don't have your own):
   ```bash
   python data_generation.py --n_series 1 --length 1000 --save results/data.npy
   ```
3. Train baseline (LSTM) with rolling-origin CV:
   ```bash
   python train.py --model baseline --data results/data.npy --out results/baseline
   ```
4. Train transformer:
   ```bash
   python train.py --model transformer --data results/data.npy --out results/transformer
   ```
5. Analyze attention and open REPORT.md

## Files
- `data_generation.py` : create synthetic demo data (or replace with your dataset loader)
- `model.py` : BaselineLSTM and TimeSeriesTransformer (with positional encoding)
- `train.py` : improved training loop + rolling-origin CV + metrics + checkpointing
- `metrics.py` : MAE, RMSE, MAPE
- `utils_cv.py` : rolling-origin split helpers
- `interpret_attention.py` : utilities to plot attention heatmaps
- `REPORT.md` : template to fill for submission
