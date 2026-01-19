import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import warnings
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast
from sklearn.metrics import r2_score



from models.model import *
from utils.dataset import *
from utils.scores import *


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

enc_in       = 100
dec_in       = 1
c_out        = 1
seq_len      = 96
label_len    = 24
out_len      = 1
factor       = 5       
d_model      = 128
n_heads      = 4
e_layers     = 2
d_layers     = 1
d_ff         = 512
dropout      = 0.1
attn         = 'full'  
embed        = 'timeF'
freq         = 'd'
activation   = 'gelu'
distil       = False    
mix          = True
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FOLDER  = "/kaggle/working/Market-Data"  
SEQ_LEN      = 96
LABEL_LEN    = 48
PRED_LEN     = 1
STRIDE       = 96//2 
TARGET_COL   = "% Change (Pred)"
batch_size   = 64

train_dataset = InformerDataset(
    data_folder=DATA_FOLDER,
    split="train",
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    stride=STRIDE,
    target_col=TARGET_COL
)

print("Train samples:", len(train_dataset))

val_dataset = InformerDataset(
    data_folder=DATA_FOLDER,
    split="val",
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    stride=STRIDE,
    target_col=TARGET_COL,
    feature_schema=train_dataset.feature_schema  
)

print("Validation samples:", len(val_dataset))


test_dataset = InformerDataset(
    data_folder=DATA_FOLDER,
    split="test",
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    stride=STRIDE,
    target_col=TARGET_COL,
    feature_schema=train_dataset.feature_schema  
)

print("Test samples:", len(test_dataset))


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    # persistent_workers = True, prefetch_factor = 4
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    # persistent_workers = True, prefetch_factor = 4
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    # persistent_workers = True, prefetch_factor = 4
)


model = Informer(
    enc_in=enc_in,
    dec_in=dec_in,
    c_out=c_out,
    seq_len=seq_len,
    label_len=label_len,
    out_len=out_len,
    factor=factor,
    d_model=d_model,
    n_heads=n_heads,
    e_layers=e_layers,
    d_layers=d_layers,
    d_ff=d_ff,
    dropout=dropout,
    attn=attn,
    embed=embed,
    freq=freq,
    activation=activation,
    distil=True,
    mix=True,
    device=device
)
model.to(device)
print(f'model loaded into {device} sucessfully')

weight_decay = 1e-4
LR           = 1e-4
min_lr       = 1e-6
num_epochs   = 42
seed         = 42
use_amp      = True
max_steps    = num_epochs * len(train_loader)
warmup_steps = 0.01*max_steps
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler       = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
loss_fn      = torch.nn.HuberLoss(delta=1.0) 
grad_clip    = 5.0
path         = '/kaggle/working/Informer-logs'

final_train_loss = []
final_valid_loss = []
final_grad_logs  = []
final_mae        = []
final_mse        = []
final_rmse       = []
final_mape       = []
final_mapse      = []
final_r2         = []


config = {
    "model": "Informer",
    "seq_len": seq_len,
    "label_len": label_len,
    "pred_len": out_len,
    "enc_in": enc_in,
    "dec_in": dec_in,
    "c_out": c_out,
    "d_model": d_model,
    "n_heads": n_heads,
    "e_layers": e_layers,
    "d_layers": d_layers,
    "d_ff": d_ff,
    "dropout": dropout,
    "attn": "prob",
    "loss": "MSE",
    "optimizer": "AdamW",
    "lr": optimizer.param_groups[0]["lr"],
    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    "grad_clip": grad_clip,
    "seed": seed,
}

def save_checkpoint(
    save_path: str,
    model,
    optimizer,
    epoch: int,
    scheduler=None,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,

        # "global_step": global_step,=
        "config": config,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "save_time": datetime.datetime.now().isoformat()
    }
    torch.save(checkpoint, save_path)


def appending(train_val, grad, materics):
    final_train_loss.append(train_val)
    final_valid_loss.append(materics['val_loss'])
    final_grad_logs.append(grad)
    final_mae.append(materics['mae'])
    final_mse.append(materics['mse'])
    final_rmse.append(materics['rmse'])
    final_mape.append(materics['mape'])
    final_mapse.append(materics['mapse'])
    final_r2.append(materics['r2_score'])
    

def ploting(path,epoch):
    def avg_loss_ploting():
      plt.plot(final_train_loss, label='Loss')
      plt.xlabel('epochs')
      plt.ylabel('Loss')
      plt.title(f'Loss vs epochs with batch_size {batch_size}')
      plt.savefig(f'{path}/training_loss.png')
      plt.close()
        
    def grad_ploting():
      plt.plot(final_grad_logs, label='Loss')
      plt.xlabel('epochs')
      plt.ylabel('Gradient')
      plt.title(f'gradient vs epochs with batch_size {batch_size}')
      plt.savefig(f'{path}/gradient_flow.png')
      plt.close()
    def train_valid():
        epochs = np.arange(len(final_train_loss))
        if len(final_train_loss) == 0 or len(final_valid_loss) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, final_train_loss,
                 marker='o', linewidth=2,
                 label='Train Loss')
        
        plt.plot(epochs, final_valid_loss,
                 marker='s', linewidth=2,
                 linestyle='--',
                 label='Validation Loss')
        
        # Styling
        plt.title("Training vs Validation Loss", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{path}/train_valid_plot.png')
        plt.close()
    def valid_ploting():
        epochs = [i for i in range(1,epoch + 1)]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("Validation Metrics History", fontsize=16)
    
        # ---- MAE ----
        axes[0, 0].plot(epochs, final_mae, marker='o')
        axes[0, 0].set_title("MAE (Mean Absolute Error)")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Error")
        axes[0, 0].grid(True)
        axes[0, 0].legend(["MAE"])
    
        # ---- MSE ----
        axes[0, 1].plot(epochs, final_mse, marker='o')
        axes[0, 1].set_title("MSE (Mean Squared Error)")
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Error Squared")
        axes[0, 1].grid(True)
        axes[0, 1].legend(["MSE"])
    
        # ---- RMSE ----
        axes[1, 0].plot(epochs, final_rmse, marker='o')
        axes[1, 0].set_title("RMSE (Root Mean Squared Error)")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Error")
        axes[1, 0].grid(True)
        axes[1, 0].legend(["RMSE"])
    
        # ---- MAPE ----
        axes[1, 1].plot(epochs, final_mape, marker='o')
        axes[1, 1].set_title("MAPE (Mean Absolute Percentage Error)")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Percentage Error")
        axes[1, 1].grid(True)
        axes[1, 1].legend(["MAPE"])
    
        # ---- MAPSE ----
        axes[2, 0].plot(epochs, final_mapse, marker='o')
        axes[2, 0].set_title("MAPSE (Mean Absolute Percentage Squared Error)")
        axes[2, 0].set_xlabel("Epochs")
        axes[2, 0].set_ylabel("Percentage Squared Error")
        axes[2, 0].grid(True)
        axes[2, 0].legend(["MAPSE"])
    
        # ---- R2 ----
        axes[2, 1].plot(epochs, final_r2, marker='o')
        axes[2, 1].set_title("R² Score")
        axes[2, 1].set_xlabel("Epochs")
        axes[2, 1].set_ylabel("Score")
        axes[2, 1].grid(True)
        axes[2, 1].legend(["R²"])
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{path}/valid_plot.png')
        plt.close()
    avg_loss_ploting()
    grad_ploting()
    train_valid()
    valid_ploting()


def evaluation(model, valid_loader):
    model.eval()
    mae_list, mse_list, rmse_list, mape_list, mspe_list, r2_list, loss_list = [],[],[],[],[],[], []
    
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            x_enc = batch['x_enc'].to(device)
            x_enc_mark = batch['x_mark_enc'].to(device)
            x_dec = batch['x_dec'].to(device)
            x_dec_mark = batch['x_mark_dec'].to(device)
            y_true = batch['y_true'].to(device)
            
           
            outputs = model(x_enc, x_enc_mark, x_dec, x_dec_mark)
            
            if outputs.shape[-1] != 1: outputs = outputs[:, :, -1:]
            
            batch_mean = batch['tgt_mean'].to(device).view(-1, 1, 1) 
            batch_std  = batch['tgt_std'].to(device).view(-1, 1, 1)  
            
            outputs_real = (outputs * batch_std) + batch_mean
            y_true_real  = (y_true * batch_std) + batch_mean
            
            loss = loss_fn(outputs_real, y_true_real)
            loss_list.append(loss.item())
            
            pred_np = outputs_real.detach().cpu().numpy().flatten()
            true_np = y_true_real.detach().cpu().numpy().flatten()
            
            # 7. Calculate Metrics
            matrices = metric(pred_np, true_np)
            
            mae_list.append(matrices['MAE'])
            mse_list.append(matrices['MSE'])
            rmse_list.append(matrices['RMSE'])
            mape_list.append(matrices['MAPE'])
            mspe_list.append(matrices['MSPE'])
            r2_list.append(matrices['R2_Score'])
            
            # --- DEBUG ONE BATCH ---
            # if i == 0:
            #     print(f"\n[DEBUG BATCH 0]")
            #     print(f"Norm Output (Mean): {outputs.mean().item():.4f}")
            #     print(f"Real Output (Mean): {outputs_real.mean().item():.4f}")
            #     print(f"Real Target (Mean): {y_true_real.mean().item():.4f}")
            #     print(f"Target Mean (Example): {batch_mean[0].item():.4f}")
            #     print(f"Target Std  (Example): {batch_std[0].item():.4f}")

 
    print(f'Validation || Loss: {np.mean(loss_list):.3f} || MAE: {np.mean(mae_list):.3f} || R2: {np.mean(r2_list):.3f}')
    
    return {
        'val_loss' : np.mean(loss_list),
        'mae': np.mean(mae_list),
        'mse': np.mean(mse_list),
        'rmse': np.mean(rmse_list),
        'mape': np.mean(mape_list),
        'mapse': np.mean(mspe_list),
        'r2_score': np.mean(r2_list),
    }


def train(path):
    print(f'Training started.........')
    set_seed(seed)
    max_nan_batches = 5
    for epoch in range(1,num_epochs +1):
        model.train()
        avg_loss = []
        avg_gradient = 0
        nan_batches = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            x_enc      = batch['x_enc'].to(device)
            x_mark_enc = batch['x_mark_enc'].to(device)
            x_dec      = batch['x_dec'].to(device)
            x_mark_dec = batch['x_mark_dec'].to(device)
            y_true     = batch['y_true'].to(device)
            output     = model(x_enc,x_mark_enc.float(), x_dec,x_mark_dec.float())

            loss = loss_fn(output, y_true)
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches > max_nan_batches:
                    raise RuntimeError(
                        f"Training unstable: {nan_batches} NaN/Inf batches"
                    )
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip)
            if not torch.isfinite(grad_norm):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches > max_nan_batches:
                    raise RuntimeError(
                        f"Training unstable: {nan_batches} NaN/Inf gradients"
                    )
                continue
            optimizer.step()
            # scheduler.step()
            avg_loss.append(loss.item())
            avg_gradient += grad_norm.item()
            if step== len(train_loader)-1:
                print(f'Epoch: {epoch}/{num_epochs} || Loss: {loss.item():.4f} || Gradient: {grad_norm:.4f} || avg_loss: {np.mean(avg_loss):.4f} ||')
        
        train_loss = np.mean(avg_loss)
        scheduler.step(train_loss)
        matrices = evaluation(model,val_loader)   
        appending(train_loss, avg_gradient/len(train_loader), matrices)
        ploting(path,epoch) 
        save_checkpoint(f'{path}/model.pt',model,optimizer,epoch)


train(path)
