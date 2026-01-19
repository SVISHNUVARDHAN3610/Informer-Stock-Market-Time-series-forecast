import torch
import pandas as pd
import numpy as np
import os

CONFIG = {
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 1,
    'target_col': "% Change (Pred)", 
    'model_path': "informer_stock_success.pth",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def clean_numeric_df(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df

def extract_time_features(dates: pd.Series):
    return np.stack([
        dates.dt.month.values,
        dates.dt.day.values,
        dates.dt.weekday.values,
        dates.dt.dayofyear.values,
    ], axis=1)

def predict_next_move(csv_path, model):
    
    if not os.path.exists(csv_path):
        return f"Error: File {csv_path} not found."
    
    df = pd.read_csv(csv_path)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date")
    else:
        df["Date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    

    if len(df) < CONFIG['seq_len']:
        return f"Error: Not enough data. Need at least {CONFIG['seq_len']} rows, got {len(df)}."


    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    
    if CONFIG['target_col'] not in numeric_cols:
        return f"Error: Target column '{CONFIG['target_col']}' not found in CSV."
    
    feature_cols = [c for c in numeric_cols if c != CONFIG['target_col']]
    
  
    feat_data = clean_numeric_df(df[feature_cols]).values
    feat_mean = feat_data.mean(axis=0)
    feat_std = feat_data.std(axis=0) + 1e-6
    

    target_data = clean_numeric_df(df[[CONFIG['target_col']]]).values
    target_mean = target_data.mean()
    target_std = target_data.std() + 1e-6
    
    last_seq_feat = feat_data[-CONFIG['seq_len']:] 
    last_seq_target = target_data[-CONFIG['seq_len']:]
    

    norm_feat = (last_seq_feat - feat_mean) / feat_std
    norm_target = (last_seq_target - target_mean) / target_std
    
  
    x_enc = torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
    

    dates = df["Date"].iloc[-CONFIG['seq_len'] - CONFIG['pred_len']:]
    time_feats = extract_time_features(dates)
    last_date = df["Date"].iloc[-1]
    future_date = last_date + pd.Timedelta(days=1)
    

    t_enc = torch.tensor(time_feats[-CONFIG['seq_len']:], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
    

    dec_context = norm_target[-CONFIG['label_len']:]
    last_val = norm_target[-1]
    placeholder = np.full((CONFIG['pred_len'], 1), last_val)
    
    x_dec_input = np.vstack([dec_context, placeholder])
    x_dec = torch.tensor(x_dec_input, dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
    
    t_dec = torch.tensor(time_feats[-CONFIG['label_len']-CONFIG['pred_len']:], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])

    model.eval()
    with torch.no_grad():
        output = model(x_enc, t_enc, x_dec, t_dec)
        if output.shape[-1] != 1: output = output[:, :, -1:]
        
        pred_norm = output.item()
        
    pred_raw = (pred_norm * target_std) + target_mean
    
    pred_return = pred_raw / 100.0
    
    threshold = 0.0005
    
    print(f"\n--- PREDICTION REPORT for {os.path.basename(csv_path)} ---")
    print(f"Predicted Raw Value: {pred_raw:.4f}")
    print(f"Predicted Return:    {pred_return*100:.4f}%")
    
    if pred_return > threshold:
        return "🟢 BUY SIGNAL (Long)"
    elif pred_return < -threshold:
        return "🔴 SELL SIGNAL (Short)"
    else:
        return "⚪ HOLD / FLAT"
