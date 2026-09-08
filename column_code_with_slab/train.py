#!/usr/bin/env python
# coding: utf-8

# ----------------- config -----------------
N_TRIALS = 10000
exp_name = 'gmd_radiative_convective_1420w_10min_v3'
exp_folder_name = 'column_code_with_slab'
emulate = 'conv'              # 'rad' or 'conv'
emulate_layer = 28
model_name = 'trsfm_v2_Tw_5'       # 'mlp' or 'mlp_v2' or 'trsfm_v2'
ml_exp_name = 'best_model' + '_' + emulate + '_' + model_name

if emulate == 'rad':
    IN_FEATURES, OUT_FEATURES = 59, 28
if emulate == 'conv':
    IN_FEATURES, OUT_FEATURES = 59 - (28 - emulate_layer) * 2, 57 - (28 - emulate_layer) * 2

EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
SCHEDULER_PATIENCE = 10

# ---- hard constraint settings ----
# OUTPUT: zero dT/dQ above this level (levels >= CLAMP_FROM_LEVEL)
CLAMP_FROM_LEVEL = 19     # must be in [0, emulate_layer]

# INPUT: zero T/Q inputs above this level (levels >= INPUT_CLAMP_FROM_LEVEL)
# This prevents top inputs from influencing bottom predictions.
INPUT_CLAMP_FROM_LEVEL = 19  # usually set equal to CLAMP_FROM_LEVEL

NLEV = emulate_layer      # number of vertical levels in T/Q and dT/dQ blocks

# ----------------- imports -----------------
import os, sys, random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

sys.path.append("..")
from models import (
    DynamicMLP_flatten,
    DynamicMLP_flatten_v2,
    WindowTemporalTransformer_v2,   # forward(x); has input_clamp_from_level + clamp_from_level
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(ml_exp_name, exist_ok=True)

# ----------------- load data -----------------
X_train = torch.load(
    f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/X_train.pth",
    weights_only=True
)
X_val = torch.load(
    f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/X_val.pth",
    weights_only=True
)
y_train = torch.load(
    f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/y_train.pth",
    weights_only=True
)
y_val = torch.load(
    f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/y_val.pth",
    weights_only=True
)
norm = torch.load(
    f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/normalization.pth",
    weights_only=False
)

X_max = norm['X_max']
X_min = norm['X_min']
y_mean = norm['y_mean']
y_std  = norm['y_std']

X_den = (X_max - X_min).clone()
X_den[X_den == 0.0] = 1.0
y_std_safe = y_std.clone()
y_std_safe[y_std_safe == 0.0] = 1.0

# normalize
X_train_n = (X_train - X_min) / X_den
X_val_n   = (X_val   - X_min) / X_den
y_train_n = (y_train - y_mean) / y_std_safe
y_val_n   = (y_val   - y_mean) / y_std_safe

# ----------------- windows helper -----------------
def make_windows(Xn, yn, Xraw, yraw, Tw):
    """
    Xn:   [N, C]
    yn:   [N, O]
    Xraw: [N, C]
    yraw: [N, O]
    returns:
      Xn_w:   [N2, Tw, C]
      yn_w:   [N2, O]
      Xraw_w: [N2, Tw, C]
      yraw_w: [N2, O]
    """
    if Tw < 1:
        raise ValueError(f"Tw must be >= 1, got {Tw}")
    if Xn.shape[0] < Tw:
        raise ValueError(f"N={Xn.shape[0]} is smaller than Tw={Tw}")

    Xn_w   = Xn.unfold(0, Tw, 1).permute(0, 2, 1).contiguous()
    Xraw_w = Xraw.unfold(0, Tw, 1).permute(0, 2, 1).contiguous()

    yn_w   = yn[Tw - 1:].contiguous()
    yraw_w = yraw[Tw - 1:].contiguous()

    return Xn_w, yn_w, Xraw_w, yraw_w

# Cache windowed datasets by Tw so we don't rebuild every trial
WINDOW_CACHE = {}  # key: Tw -> (training_set, val_set)

def get_windowed_sets(Tw: int):
    if Tw in WINDOW_CACHE:
        return WINDOW_CACHE[Tw]

    Xw_train_n, yw_train_n, Xw_train, yw_train = make_windows(X_train_n, y_train_n, X_train, y_train, Tw)
    Xw_val_n,   yw_val_n,   Xw_val,   yw_val   = make_windows(X_val_n,   y_val_n,   X_val,   y_val,   Tw)

    training_set = TensorDataset(Xw_train_n, yw_train_n, Xw_train, yw_train)
    val_set      = TensorDataset(Xw_val_n,   yw_val_n,   Xw_val,   yw_val)

    WINDOW_CACHE[Tw] = (training_set, val_set)
    return training_set, val_set

# ----------------- objective -----------------
def objective(trial):
    seed = 42 + trial.number
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # ----- hyperparams -----
    if model_name in ('mlp', 'mlp_v2'):
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        n_layers = trial.suggest_int("n_layers", 2, 8)
        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
        hidden_sizes = [trial.suggest_int(f"hidden_{i}", 128, 1024, step=128) for i in range(n_layers)]
        batch_size = trial.suggest_categorical("batch_size", [1024])

        training_set = TensorDataset(X_train_n, y_train_n, X_train, y_train)
        val_set      = TensorDataset(X_val_n,   y_val_n,   X_val,   y_val)
        Tw_used = 1

        # dummy placeholders so payload code doesn't error
        d_model = nhead = dim_feedforward = num_layers = dropout = None

    else:  # trsfm_v2 (input + output hard constraints)
        lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
        optimizer_name = "AdamW"
        batch_size = trial.suggest_categorical("batch_size", [1024])

        T_window = trial.suggest_categorical("T_window", [5]) # [5, 10, 15, 20]
        Tw_used = int(T_window)

        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        nhead = trial.suggest_categorical("nhead", [4])
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
        num_layers = trial.suggest_categorical("num_layers", [2, 4, 6])
        dropout = trial.suggest_categorical("dropout", [0.0])

        training_set, val_set = get_windowed_sets(Tw_used)

        # hidden_sizes not used for transformer; keep defined for safety
        hidden_sizes = None

    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dataloader   = DataLoader(val_set,      batch_size=batch_size, shuffle=False, drop_last=False)

    # ----- model -----
    if model_name == 'mlp':
        model = DynamicMLP_flatten(IN_FEATURES, OUT_FEATURES, hidden_sizes).to(device)
    elif model_name == 'mlp_v2':
        model = DynamicMLP_flatten_v2(IN_FEATURES, OUT_FEATURES, hidden_sizes).to(device)
    else:  # trsfm_v2
        model = WindowTemporalTransformer_v2(
            C_in=IN_FEATURES,
            C_out=OUT_FEATURES,
            T_window=Tw_used,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            causal=True,
            nlev=NLEV,
            clamp_from_level=CLAMP_FROM_LEVEL,            # OUTPUT clamp
            input_clamp_from_level=INPUT_CLAMP_FROM_LEVEL # INPUT mask (prevents top influence)
        ).to(device)

    # ----- losses -----
    loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")
    loss_fn_raw = nn.MSELoss()

    # ----- optimizer / scheduler -----
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)

    # ----- low-level weighting -----
    L = emulate_layer
    LOW_K = 5
    LOW_W = 10.0
    LOW_K_eff = min(LOW_K, L)

    w = torch.ones(OUT_FEATURES, device=device)
    if emulate == "conv":
        w[0:LOW_K_eff] *= LOW_W
        w[L:L + LOW_K_eff] *= LOW_W

    # ----- training loop -----
    best_val_loss_raw = float("inf")
    best_epoch = None
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_t = 0.0
        nb = 0

        for Xn, yn, Xraw, yraw in train_dataloader:
            Xn = Xn.to(device)
            yn = yn.to(device)

            optimizer.zero_grad()

            # mlp/mlp_v2: Xn is [B,C]
            # trsfm_v2:   Xn is [B,Tw,C] (input mask + output clamp inside model)
            y_pred_n = model(Xn)

            err = loss_fn(y_pred_n, yn)  # [B,O]
            loss = (err * w).mean()

            loss.backward()
            optimizer.step()

            train_loss_t += loss.detach()
            nb += 1

        train_loss = (train_loss_t / max(nb, 1)).item()

        # ---- validation in RAW units ----
        model.eval()
        val_loss_raw_t = 0.0
        with torch.no_grad():
            for Xn, yn, Xraw, yraw in val_dataloader:
                Xn = Xn.to(device)
                yraw = yraw.to(device)

                y_pred_n = model(Xn)
                y_pred = y_pred_n * y_std_safe.to(device) + y_mean.to(device)
                val_loss_raw_t += loss_fn_raw(y_pred, yraw).detach()

        val_loss_raw = (val_loss_raw_t / max(len(val_dataloader), 1)).item()

        scheduler.step(val_loss_raw)
        # print(f"epoch={epoch:03d} train={train_loss:.6g} val_raw={val_loss_raw:.6g}")

        if val_loss_raw < best_val_loss_raw:
            best_val_loss_raw = val_loss_raw
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                break

        trial.report(val_loss_raw, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ----- save trial -----
    if model_name in ('mlp', 'mlp_v2'):
        payload = {
            "model_state": best_state_dict,
            "hidden_sizes": hidden_sizes,
            "lr": lr,
            "optimizer_name": optimizer_name,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_val_loss_raw": best_val_loss_raw,
            "exp_name": exp_name,
            "exp_folder_name": exp_folder_name,
            "emulate": emulate,
            "emulate_layer": emulate_layer,
        }
    else:
        payload = {
            "model_state": best_state_dict,
            "T_window": Tw_used,
            "lr": lr,
            "optimizer_name": optimizer_name,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_val_loss_raw": best_val_loss_raw,
            "exp_name": exp_name,
            "exp_folder_name": exp_folder_name,
            "emulate": emulate,
            "emulate_layer": emulate_layer,
            "clamp_from_level": CLAMP_FROM_LEVEL,
            "input_clamp_from_level": INPUT_CLAMP_FROM_LEVEL,
            "nlev": NLEV,
        }

    torch.save(payload, f"./{ml_exp_name}/trial_{trial.number}.pth")

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss_raw", best_val_loss_raw)

    print(f"trial={trial.number} Tw={Tw_used} best_epoch={best_epoch} best_raw_val_MSE={best_val_loss_raw}")
    return best_val_loss_raw

# ----------------- run optuna -----------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)


