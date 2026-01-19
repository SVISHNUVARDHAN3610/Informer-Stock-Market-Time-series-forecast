import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def extract_time_features_daily(dates: pd.Series):
    """
    Returns: (L, 3)
    [month, day, weekday]
    """
    dates = pd.to_datetime(dates)
    return np.stack([
        dates.dt.month.values,
        dates.dt.day.values,
        dates.dt.weekday.values
    ], axis=1)



class InformerDataset(Dataset):
    

    def __init__(
        self,
        data_folder,
        split,                     # train | val | test
        seq_len,
        label_len,
        pred_len,
        stride,
        target_col,
        train_ratio=0.7,
        val_ratio=0.2,
        feature_schema=None
    ):
        assert split in ["train", "val", "test"]

        self.data_folder = data_folder
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.target_col = target_col
        self.feature_schema = feature_schema

        self.samples = []
        self._build(train_ratio, val_ratio)

    # ---------------------
    def _build(self, train_ratio, val_ratio):
        csv_files = sorted(f for f in os.listdir(self.data_folder) if f.endswith(".csv"))

        for fname in csv_files:
            df = pd.read_csv(os.path.join(self.data_folder, fname))

            if "Date" not in df.columns or self.target_col not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").drop_duplicates("Date")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

            split_df = {
                "train": train_df,
                "val": val_df,
                "test": test_df
            }[self.split]

            if len(split_df) < self.seq_len + self.pred_len:
                continue

            if self.feature_schema is None:
                self.feature_schema = (
                    train_df
                    .drop(columns=["Date", self.target_col])
                    .select_dtypes(include=[np.number])
                    .columns
                    .tolist()
                )

       
            feat_train = train_df[self.feature_schema].values
            feat_mean = feat_train.mean(axis=0)
            feat_std = feat_train.std(axis=0) + 1e-6

            tgt_train = train_df[self.target_col].values
            tgt_mean = tgt_train.mean()
            tgt_std = tgt_train.std() + 1e-6

            feat_all = (split_df[self.feature_schema].values - feat_mean) / feat_std
            tgt_all = (split_df[self.target_col].values - tgt_mean) / tgt_std
            tgt_all = tgt_all.reshape(-1, 1)

            time_feats = extract_time_features_daily(split_df["Date"])

            total_len = self.seq_len + self.pred_len

            for i in range(0, len(split_df) - total_len + 1, self.stride):
                enc_end = i + self.seq_len
                dec_start = enc_end - self.label_len
                dec_end = enc_end + self.pred_len

                x_enc = feat_all[i:enc_end]
                x_mark_enc = time_feats[i:enc_end]

                past_y = tgt_all[dec_start:enc_end]
                future_zeros = np.zeros((self.pred_len, 1))
                x_dec = np.vstack([past_y, future_zeros])

                x_mark_dec = time_feats[dec_start:dec_end]
                y_true = tgt_all[enc_end:enc_end + self.pred_len]

                self.samples.append({
                    "x_enc": torch.tensor(x_enc, dtype=torch.float32),
                    "x_mark_enc": torch.tensor(x_mark_enc, dtype=torch.long),
                    "x_dec": torch.tensor(x_dec, dtype=torch.float32),
                    "x_mark_dec": torch.tensor(x_mark_dec, dtype=torch.long),
                    "y_true": torch.tensor(y_true, dtype=torch.float32),

                    # 🔥 REQUIRED FOR DE-NORMALIZATION
                    "feat_mean": torch.tensor(feat_mean, dtype=torch.float32),
                    "feat_std": torch.tensor(feat_std, dtype=torch.float32),
                    "tgt_mean": torch.tensor(tgt_mean, dtype=torch.float32),
                    "tgt_std": torch.tensor(tgt_std, dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
