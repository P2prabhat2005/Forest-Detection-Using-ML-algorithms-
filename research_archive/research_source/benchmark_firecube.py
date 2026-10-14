"""FWI vs XGBoost vs LSTM on leakage-safe FireCube samples. No extra architectures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data" / "processed" / "firecube_samples.npz"
OUT = ROOT / "results" / "firecube_benchmark"
SEED = 1234
H_MAIN = 1
LEAD_TIMES = (1, 3, 5, 7)
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
EPOCHS = 25
BATCH = 256
HIDDEN = 32
LR = 1e-3


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def window_slice(history: int, h: int):
    """buffer index 0 is t-history, index history-1 is t-1.
    Lead H uses 10 days ending at t-H: t-(H+9) .. t-H."""
    last = history - h
    start = last - 9
    end = last + 1
    if start < 0:
        raise ValueError(f"history={history} too short for H={h}")
    return start, end


def fill_time_nans(dyn: np.ndarray) -> np.ndarray:
    """GRL-style: replace NaNs with per-sample column mean, then -1."""
    out = dyn.copy()
    n, t, c = out.shape
    for i in range(n):
        colmean = np.nanmean(out[i], axis=0)
        colmean = np.where(np.isfinite(colmean), colmean, 0.0)
        inds = np.where(~np.isfinite(out[i]))
        out[i][inds] = colmean[inds[1]]
    return np.nan_to_num(out, nan=-1.0)


def split_mask(year, name):
    if name == "train":
        return year <= 2018
    if name == "val":
        return year == 2019
    if name == "test2020":
        return year == 2020
    if name == "test2021":
        return year == 2021
    raise ValueError(name)


def far_fpr(y_true, y_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    far_among_alarms = fp / (fp + tp) if (fp + tp) else 0.0
    return fpr, far_among_alarms, int(tn), int(fp), int(fn), int(tp)


def metrics_at(y_true, p, thr):
    y_hat = (p >= thr).astype(np.int32)
    fpr, far_al, tn, fp, fn, tp = far_fpr(y_true, y_hat)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "fpr": float(fpr),
        "false_alarm_among_alerts": float(far_al),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def ranking_metrics(y_true, p):
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "base_rate": float(y_true.mean()),
    }


def best_f1_threshold(y_true, p, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 37)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y_true, p >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, float(best_f)


def xgb_features(dyn10, static, clc):
    last = dyn10[:, -1, :]
    mean = dyn10.mean(axis=1)
    return np.concatenate([mean, last, static, clc], axis=1)


class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        x = self.ln(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm(x_train, y_train, x_val, y_val, device):
    model = SimpleLSTM(x_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    pos = y_train.mean()
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_np = y_val
    best_state, best_pr, wait = None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x_val_t)).cpu().numpy()
        pr = average_precision_score(y_val_np, p)
        print(f"  LSTM epoch {epoch+1}/{EPOCHS} loss={tot/len(y_train):.4f} val_PR={pr:.4f}", flush=True)
        if pr > best_pr:
            best_pr, wait = pr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                print("  early stop", flush=True)
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def lstm_predict(model, x, device):
    model.eval()
    out = []
    xt = torch.from_numpy(x)
    for i in range(0, len(x), 2048):
        logits = model(xt[i : i + 2048].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def evaluate_model(name, p_val, y_val, tests, fwi_val=None):
    if name == "FWI":
        # calibrate FWI to [0,1] via empirical CDF on train/val scores already passed as p
        thr, _ = best_f1_threshold(y_val, p_val)
    else:
        thr, _ = best_f1_threshold(y_val, p_val)
    result = {"val_best_f1_threshold": thr, "val": {**ranking_metrics(y_val, p_val), **metrics_at(y_val, p_val, thr)}}
    result["val"]["threshold_sweep"] = [metrics_at(y_val, p_val, t) for t in THRESHOLDS]
    for split, (y, p) in tests.items():
        block = ranking_metrics(y, p)
        block["at_val_best_f1"] = metrics_at(y, p, thr)
        block["at_0.5"] = metrics_at(y, p, 0.5)
        block["threshold_sweep"] = [metrics_at(y, p, t) for t in THRESHOLDS]
        result[split] = block
    return result


def main():
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ, allow_pickle=True)
    dyn_raw = data["dynamic"]
    fwi_raw = data["fwi"]
    static = np.nan_to_num(data["static"], nan=-1.0).astype(np.float32)
    clc = np.nan_to_num(data["clc"], nan=0.0).astype(np.float32)
    label = data["label"].astype(np.float32)
    year = data["year"]
    history = dyn_raw.shape[1]
    print("loaded", dyn_raw.shape, "pos", int(label.sum()), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    train_m = split_mask(year, "train")
    val_m = split_mask(year, "val")
    t20_m = split_mask(year, "test2020")
    t21_m = split_mask(year, "test2021")

    # --- H=1 windows ---
    s, e = window_slice(history, H_MAIN)
    dyn = fill_time_nans(dyn_raw[:, s:e, :])
    fwi_w = np.nan_to_num(fwi_raw[:, s:e], nan=-1.0)
    fwi_score_raw = fwi_w[:, -1]  # FWI at t-1

    scaler_dyn = MinMaxScaler()
    n, t, c = dyn.shape
    scaler_dyn.fit(dyn[train_m].reshape(-1, c))
    dyn_n = scaler_dyn.transform(dyn.reshape(-1, c)).reshape(n, t, c).astype(np.float32)
    scaler_sta = MinMaxScaler().fit(static[train_m])
    sta_n = scaler_sta.transform(static).astype(np.float32)

    xgb_x = xgb_features(dyn_n, sta_n, clc)
    lstm_x = np.concatenate(
        [dyn_n, np.repeat(sta_n[:, None, :], t, axis=1), np.repeat(clc[:, None, :], t, axis=1)],
        axis=2,
    ).astype(np.float32)

    # FWI as probability: min-max on train then clip
    fwi_min, fwi_max = np.percentile(fwi_score_raw[train_m], [1, 99])
    fwi_p = np.clip((fwi_score_raw - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1).astype(np.float32)

    print("Training XGBoost...", flush=True)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=4,
        random_state=SEED,
        early_stopping_rounds=40,
    )
    xgb.fit(
        xgb_x[train_m],
        label[train_m],
        eval_set=[(xgb_x[val_m], label[val_m])],
        verbose=False,
    )
    p_xgb = {
        "val": xgb.predict_proba(xgb_x[val_m])[:, 1],
        "test2020": xgb.predict_proba(xgb_x[t20_m])[:, 1],
        "test2021": xgb.predict_proba(xgb_x[t21_m])[:, 1],
    }

    print("Training LSTM...", flush=True)
    lstm = train_lstm(lstm_x[train_m], label[train_m], lstm_x[val_m], label[val_m], device)
    torch.save(lstm.state_dict(), OUT / "lstm.pt")
    p_lstm = {
        "val": lstm_predict(lstm, lstm_x[val_m], device),
        "test2020": lstm_predict(lstm, lstm_x[t20_m], device),
        "test2021": lstm_predict(lstm, lstm_x[t21_m], device),
    }

    y_val = label[val_m]
    tests_fwi = {"test2020": (label[t20_m], fwi_p[t20_m]), "test2021": (label[t21_m], fwi_p[t21_m])}
    tests_xgb = {"test2020": (label[t20_m], p_xgb["test2020"]), "test2021": (label[t21_m], p_xgb["test2021"])}
    tests_lstm = {"test2020": (label[t20_m], p_lstm["test2020"]), "test2021": (label[t21_m], p_lstm["test2021"])}

    report = {
        "protocol": {
            "target": "burned_areas cell on start day t (EFFIS raster)",
            "features": "dynamic t-10..t-1; static DEM/slope/roads/water/pop; CLC 10-frac",
            "fwi": "FWI at t-1, 1-99 percentile scaled on train",
            "xgb": "mean(10d)+last day+static+CLC",
            "lstm": "10-step sequence, LayerNorm+LSTM32+MLP, BCE",
            "splits": "train 2009-2018, val 2019, test 2020 / 2021",
            "neg_pos": 2,
            "xgb_best_iteration": int(getattr(xgb, "best_iteration", -1)),
        },
        "sample_counts": {
            k: {"n": int(m.sum()), "n_pos": int(label[m].sum())}
            for k, m in [("train", train_m), ("val", val_m), ("test2020", t20_m), ("test2021", t21_m)]
        },
        "FWI": evaluate_model("FWI", fwi_p[val_m], y_val, tests_fwi),
        "XGBoost": evaluate_model("XGBoost", p_xgb["val"], y_val, tests_xgb),
        "LSTM": evaluate_model("LSTM", p_lstm["val"], y_val, tests_lstm),
    }

    print("Lead-time sweep...", flush=True)
    lead = {}
    for h in LEAD_TIMES:
        hs, he = window_slice(history, h)
        dyn_h = fill_time_nans(dyn_raw[:, hs:he, :])
        dyn_h = scaler_dyn.transform(dyn_h.reshape(-1, c)).reshape(n, 10, c).astype(np.float32)
        fwi_h = np.nan_to_num(fwi_raw[:, hs:he], nan=-1.0)[:, -1]
        fwi_ph = np.clip((fwi_h - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)
        xgb_h = xgb_features(dyn_h, sta_n, clc)
        lstm_h = np.concatenate(
            [dyn_h, np.repeat(sta_n[:, None, :], 10, axis=1), np.repeat(clc[:, None, :], 10, axis=1)],
            axis=2,
        ).astype(np.float32)
        px = xgb.predict_proba(xgb_h)[:, 1]
        pl = lstm_predict(lstm, lstm_h, device)
        lead[str(h)] = {}
        for split, m in [("val", val_m), ("test2020", t20_m), ("test2021", t21_m)]:
            y = label[m]
            lead[str(h)][split] = {
                "FWI": ranking_metrics(y, fwi_ph[m]),
                "XGBoost": ranking_metrics(y, px[m]),
                "LSTM": ranking_metrics(y, pl[m]),
            }
        print(f"  H={h} test2020 PR XGB={lead[str(h)]['test2020']['XGBoost']['pr_auc']:.3f} LSTM={lead[str(h)]['test2020']['LSTM']['pr_auc']:.3f}", flush=True)
    report["lead_time"] = lead

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT / "metrics.json", flush=True)
    for model in ("FWI", "XGBoost", "LSTM"):
        for split in ("test2020", "test2021"):
            b = report[model][split]
            op = b["at_val_best_f1"]
            print(
                f"{model:8} {split} PR={b['pr_auc']:.3f} ROC={b['roc_auc']:.3f} "
                f"F1={op['f1']:.3f} P={op['precision']:.3f} R={op['recall']:.3f} "
                f"Brier={b['brier']:.3f} FPR={op['fpr']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
