# Report template - Attention Time-Series Project

## 1. Dataset characteristics
- Number of series: 1 (demo)
- Length: TODO
- Frequency: TODO (e.g., hourly/daily)
- Features: 1 (univariate)
- Missing values: TODO
- Scaling: StandardScaler / MinMax used? TODO

## 2. Experimental setup
- Task: One-step ahead forecasting (horizon=1)
- Window/input length: TODO (e.g., 48)
- Rolling-origin CV: number of folds and sizes (see `results/cv_metrics_*.csv`)

## 3. Models
- **Baseline LSTM**: 2 layers, hidden=64, dropout=0.1
- **Transformer**: d_model=64, nhead=4, encoder_layers=2, decoder_layers=1

- Training: optimizer=Adam, lr=1e-3, scheduler=ReduceLROnPlateau, early stopping patience=7

## 4. Hyperparameters (table)
| param | value |
|---|---|
| lr | 1e-3 |
| batch_size | 64 |
| epochs | 50 |
| input_len | 48 |

## 5. Quantitative results (fill with results/cv_metrics_*.csv)
- Add table: fold, mae, rmse, mape for baseline and transformer and average row.

## 6. Visuals
- Training/validation loss curves (per fold) -> include history CSVs as plots.
- Prediction vs ground-truth plot for 3 representative sequences.
- Attention heatmaps (include figures produced by interpret_attention.py)

## 7. Attention interpretation
- Describe where attention mass concentrates (recent timesteps? seasonal lags?). Provide numbers: e.g., "mean attention on last 5 timesteps = X%".

## 8. Conclusion
- Summarize performance and propose next steps.
