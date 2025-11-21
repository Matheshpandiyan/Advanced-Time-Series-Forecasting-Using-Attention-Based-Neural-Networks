import argparse, os, time, json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from metrics import mae, rmse, mape
from utils_cv import rolling_origin_splits
from model import BaselineLSTM, TimeSeriesTransformer
import pandas as pd
import random

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataset_windows(arr, input_len=48, horizon=1):
    # arr: (T, features)
    X, Y = [], []
    T = len(arr)
    for i in range(input_len, T - horizon + 1):
        X.append(arr[i-input_len:i])
        Y.append(arr[i:i+horizon][:,-1])  # keep last feature as target
    return np.stack(X), np.stack(Y)[:,:,None]  # X:(N,seq,feat), Y:(N,horizon,1)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x).squeeze(-1)
        loss = criterion(y_pred, y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            preds.append(y_pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds).reshape(-1)
    trues = np.concatenate(trues).reshape(-1)
    return preds, trues

def run_cv(arr, model_name='baseline', out_dir='results', input_len=48, horizon=1,
           n_splits=5, epochs=50, batch_size=64, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    X, Y = create_dataset_windows(arr, input_len=input_len, horizon=horizon)
    n = len(X)
    # choose rolling CV splits
    splits = rolling_origin_splits(n, n_splits=n_splits)
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {fold+1}/{len(splits)} -- train {len(train_idx)} val {len(val_idx)}")
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(Y_val, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        if model_name == 'baseline':
            model = BaselineLSTM(input_size=X.shape[-1], hidden_size=64, num_layers=2, dropout=0.1, output_size=1)
        else:
            model = TimeSeriesTransformer(input_size=X.shape[-1], d_model=64, nhead=4,
                                          num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=128,
                                          dropout=0.1, output_size=1)
        model = model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        best_val = float('inf'); patience = 7; cur_patience = 0
        history = []

        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            preds, trues = evaluate(model, val_loader, device)
            val_rmse = np.sqrt(np.mean((preds - trues)**2))
            scheduler.step(val_rmse)
            val_mae = mae(trues, preds); val_mape = mape(trues, preds)
            history.append({'epoch':epoch,'train_loss':train_loss,'val_rmse':val_rmse,'val_mae':val_mae,'val_mape':val_mape})
            print(f"Fold{fold+1} Epoch {epoch} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f} val_mae={val_mae:.6f}")
            if val_rmse < best_val:
                best_val = val_rmse
                torch.save(model.state_dict(), os.path.join(out_dir, f"best_{model_name}_fold{fold}.pt"))
                cur_patience = 0
            else:
                cur_patience += 1
                if cur_patience >= patience:
                    print('Early stopping triggered')
                    break

        # load best model and evaluate final
        model.load_state_dict(torch.load(os.path.join(out_dir, f"best_{model_name}_fold{fold}.pt")))
        preds, trues = evaluate(model, val_loader, device)
        fold_metrics = {'fold': fold, 'mae': mae(trues, preds), 'rmse': rmse(trues, preds), 'mape': mape(trues, preds)}
        all_metrics.append(fold_metrics)
        # save fold history and predictions
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"history_{model_name}_fold{fold}.csv"), index=False)
        np.save(os.path.join(out_dir, f"preds_{model_name}_fold{fold}.npy"), preds)
        np.save(os.path.join(out_dir, f"trues_{model_name}_fold{fold}.npy"), trues)
        pd.DataFrame(all_metrics).to_csv(os.path.join(out_dir, f"cv_metrics_{model_name}.csv"), index=False)
    return pd.DataFrame(all_metrics)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='path to .npy time-series data (T, features)')
    p.add_argument('--model', type=str, default='baseline', choices=['baseline','transformer'])
    p.add_argument('--out', type=str, default='results')
    p.add_argument('--input_len', type=int, default=48)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()
    set_seed(42)
    arr = np.load(args.data)  # shape (T, features)
    df_metrics = run_cv(arr, model_name=args.model, out_dir=args.out, input_len=args.input_len,
                        horizon=args.horizon, n_splits=args.n_splits, epochs=args.epochs,
                        batch_size=args.batch_size, device=args.device)
    print('CV metrics:'); print(df_metrics)
