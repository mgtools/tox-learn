# train_coral.py
import os, argparse, yaml, csv
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from molnet import MolNet_MS             # model: forward(x, mask, env) -> (reg_pred[B,1], cls_logits[B,K], coral_logits[B,K-1])
from dataset import MolMS_Dataset        # returns (title, x_confs[M,N,F], mask[M,N], env[D], y)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def _csv_append_row(csv_path: str, row: dict):
    if not csv_path:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

# 5 bins via EPA edges on log10 LC50
def classify_toxicity_tensor(y_log10: torch.Tensor) -> torch.Tensor:
    bins = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=y_log10.device)
    return torch.bucketize(y_log10, bins)  # int in [0..4]


def coral_targets(y_int: torch.Tensor, K: int) -> torch.Tensor:
    B = y_int.size(0)
    t = torch.arange(K-1, device=y_int.device).unsqueeze(0).expand(B, -1)
    return (y_int.unsqueeze(1) > t).float()

def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    B, Km1 = logits.shape
    K = Km1 + 1
    p = torch.sigmoid(logits)
    probs = torch.zeros(B, K, device=logits.device, dtype=logits.dtype)
    probs[:, 0] = 1.0 - p[:, 0]
    for c in range(1, K-1):
        probs[:, c] = p[:, c-1] - p[:, c]
    probs[:, K-1] = p[:, K-2]
    probs.clamp_(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs

def thresholds_to_biases(thr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    thr = thr.clamp(min=eps, max=1-eps)
    return -torch.log(thr / (1.0 - thr))

def apply_coral_biases(coral_logits: torch.Tensor,
                       biases: Optional[torch.Tensor]) -> torch.Tensor:
    if biases is None:
        return coral_logits
    if biases.device != coral_logits.device:
        biases = biases.to(coral_logits.device)
    return coral_logits + biases.view(1, -1)

@torch.no_grad()
def coral_logits_to_class_probs_with_bias(coral_logits: torch.Tensor,
                                          biases: Optional[torch.Tensor] = None
                                          ) -> torch.Tensor:
    z = apply_coral_biases(coral_logits, biases)    # (B, K-1)
    return coral_logits_to_class_probs(z)


@torch.no_grad()
def _weighted_macro_f1(y_true_np, y_pred_np, K, w=None):
    if w is None:
        w = np.ones(K, dtype=np.float32)
    else:
        w = np.asarray(w, dtype=np.float32)
    f1s = []
    for c in range(K):
        mask_c = (y_true_np == c)
        tp = int(((y_pred_np == c) & mask_c).sum())
        fp = int(((y_pred_np == c) & (~mask_c)).sum())
        fn = int(((y_pred_np != c) & mask_c).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 0.0 if prec == 0.0 or rec == 0.0 else (2 * prec * rec) / (prec + rec)
        f1s.append(f1 * w[c])
    return float(np.sum(f1s) / max(np.sum(w), 1e-8))

@torch.no_grad()
def tune_coral_thresholds_f1_simple(
    logits_all: torch.Tensor,   # (B, K-1)
    y_all: torch.Tensor,        # (B,) int [0..K-1]
    thr_grid = tuple(float(x) for x in np.arange(0.30, 0.71, 0.02)),
    coord_rounds: int = 2,
    temp_grid = None            
):

    device = logits_all.device
    Km1 = logits_all.size(1)
    K   = Km1 + 1
    y_np = y_all.detach().cpu().numpy()

    def macro_f1(y_true, y_pred, K):
        f1s = []
        for c in range(K):
            tp = int(((y_pred==c) & (y_true==c)).sum())
            fp = int(((y_pred==c) & (y_true!=c)).sum())
            fn = int(((y_pred!=c) & (y_true==c)).sum())
            prec = tp / max(tp+fp, 1)
            rec  = tp / max(tp+fn, 1)
            f1   = 0.0 if prec==0.0 or rec==0.0 else 2*prec*rec/(prec+rec)
            f1s.append(f1)
        return float(np.mean(f1s))

    def preds_from_thr(z, thr_vec):
        return (torch.sigmoid(z) > thr_vec.view(1, -1)).sum(dim=1).cpu().numpy()

    def tune_at_temp(alpha: float):
        z = logits_all * float(alpha)
        thr_vec = torch.full((Km1,), 0.50, device=device)

        for _ in range(coord_rounds):
            for t in range(Km1):
                best_thr_t, best_f1_t = thr_vec[t].item(), -1.0
                for g in thr_grid:
                    trial = thr_vec.clone(); trial[t] = float(g)
                    pred  = preds_from_thr(z, trial)
                    f1    = macro_f1(y_np, pred, K)
                    if f1 > best_f1_t:
                        best_f1_t, best_thr_t = f1, float(g)
                thr_vec[t] = best_thr_t
        pred_final = preds_from_thr(z, thr_vec)
        return thr_vec, macro_f1(y_np, pred_final, K)

    if temp_grid is None:
        thr, score = tune_at_temp(1.0)
        return 1.0, thr, score
    else:
        best_alpha, best_thr, best_score = 1.0, torch.full((Km1,), 0.50, device=device), -1.0
        for a in temp_grid:
            thr, sc = tune_at_temp(a)
            if sc > best_score:
                best_score, best_thr, best_alpha = sc, thr, float(a)
        return best_alpha, best_thr, best_score

@torch.no_grad()
def gather_coral_logits_and_targets(model, device, loader):
    model.eval()
    logits_all, y_all = [], []
    for _, x, mask, env, y in loader:
        x    = x.to(device).float()
        mask = mask.to(device).bool()
        env  = env.to(device).float()
        y    = y.to(device).float().view(-1)
        yb   = classify_toxicity_tensor(y).long()
        _, _, coral_logits = model(x, mask, env)   
        logits_all.append(coral_logits)
        y_all.append(yb)
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


def make_balanced_sampler(dataset: MolMS_Dataset) -> WeightedRandomSampler:
    labels = []
    for i in range(len(dataset)):
        _, _, _, _, y = dataset[i]
        y = torch.tensor([float(y)], dtype=torch.float32)
        labels.append(int(classify_toxicity_tensor(y).item()))
    labels = torch.tensor(labels)
    counts = torch.bincount(labels).float().clamp(min=1)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

def collate_molms(batch):
    titles, xs, masks, envs, ys = zip(*batch)
    B = len(xs)
    max_M = max(x.shape[0] for x in xs)
    N = xs[0].shape[1]
    F = xs[0].shape[2]

    def pad_M(a, target_M):
        M, N_, F_ = a.shape
        if M == target_M: return a
        pad = np.zeros((target_M - M, N_, F_), dtype=a.dtype)
        return np.concatenate([a, pad], axis=0)

    def pad_M_mask(m, target_M):
        M, N_ = m.shape
        if M == target_M: return m
        pad = np.zeros((target_M - M, N_), dtype=m.dtype)
        return np.concatenate([m, pad], axis=0)

    x = torch.from_numpy(np.stack([pad_M(a.numpy() if isinstance(a, torch.Tensor) else a, max_M) for a in xs])).float()
    m = torch.from_numpy(np.stack([pad_M_mask(mm.numpy() if isinstance(mm, torch.Tensor) else mm, max_M) for mm in masks])).bool()
    e = torch.from_numpy(np.stack([ev.numpy() if isinstance(ev, torch.Tensor) else ev for ev in envs])).float()
    y = torch.tensor([float(v) for v in ys], dtype=torch.float32)
    return titles, x, m, e, y

def joint_loss(
    pred_reg, y,
    cls_logits, coral_logits, true_bins,
    w_reg=0.30, w_ce=0.30, w_coral=0.40,
):
    # regression
    loss_reg = F.mse_loss(pred_reg.view(-1), y.view(-1), reduction='mean')
    # CE
    loss_ce = F.cross_entropy(cls_logits, true_bins, reduction='mean')
    # CORAL 
    T = coral_targets(true_bins, K=coral_logits.size(1) + 1)  # (B, K-1)
    loss_coral = F.binary_cross_entropy_with_logits(coral_logits, T, reduction='mean')

    total = (w_reg*loss_reg + w_ce*loss_ce + w_coral*loss_coral)
    return total, (loss_reg, loss_ce, loss_coral)

def train_one_epoch(
    model, device, loader, optimizer,
    grad_clip: Optional[float] = 1.0,
    w_reg=0.30, w_ce=0.30, w_coral=0.40,
):
    model.train()
    n, sse = 0, 0.0
    reg_sum = ce_sum = coral_sum = 0.0

    for _, x, mask, env, y in loader:
        x    = x.to(device).float()      # (B,M,N,F)
        mask = mask.to(device).bool()    # (B,M,N)
        env  = env.to(device).float()    # (B,D)
        y    = y.to(device).float().view(-1)
        B    = x.size(0)

        y_bin = classify_toxicity_tensor(y).long()

        optimizer.zero_grad(set_to_none=True)
        pred_reg, cls_logits, coral_logits = model(x, mask, env)

        loss, (loss_reg, loss_ce, loss_coral) = joint_loss(
            pred_reg, y, cls_logits, coral_logits, y_bin,
            w_reg=w_reg, w_ce=w_ce, w_coral=w_coral
        )
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n       += B
        sse     += float(((pred_reg.view(-1) - y)**2).sum().item())
        reg_sum += float(loss_reg.item())  * B
        ce_sum  += float(loss_ce.item())   * B
        coral_sum+= float(loss_coral.item()) * B

    rmse = (sse / max(n, 1)) ** 0.5
    return rmse, reg_sum/max(n,1), ce_sum/max(n,1), coral_sum/max(n,1)

@torch.no_grad()
def eval_joint(model, device, loader,
               coral_biases: Optional[torch.Tensor] = None,
               ensemble: bool = True,
               save_csv: Optional[str] = None):
    model.eval()
    n, sse, sum_y, sum_y2 = 0, 0.0, 0.0, 0.0
    soft_preds, coral_preds, ens_preds = [], [], []
    true_bins, rows = [], []

    for titles, x, mask, env, y in loader:
        x    = x.to(device).float()
        mask = mask.to(device).bool()
        env  = env.to(device).float()
        y    = y.to(device).float().view(-1)
        B    = x.size(0)
        yb   = classify_toxicity_tensor(y).long()

        pred_reg, cls_logits, coral_logits = model(x, mask, env)

        # regression
        n += B
        sse   += float(((pred_reg.view(-1) - y)**2).sum().item())
        sum_y += float(y.sum().item())
        sum_y2+= float((y**2).sum().item())

        # decode
        soft_probs  = F.softmax(cls_logits, dim=1)                         # (B,K)
        coral_probs = coral_logits_to_class_probs_with_bias(coral_logits, coral_biases)  # (B,K)
        biased_logits = apply_coral_biases(coral_logits, coral_biases)
        coral_hard  = (torch.sigmoid(biased_logits) > 0.5).sum(dim=1)
        soft_hard   = soft_probs.argmax(dim=1)

        if ensemble:
            ens_probs = 0.5 * (soft_probs + coral_probs)
            ens_hard  = ens_probs.argmax(dim=1)
        else:
            ens_probs = coral_probs
            ens_hard  = coral_hard

        true_bins.append(yb.cpu())
        soft_preds.append(soft_hard.cpu())
        coral_preds.append(coral_hard.cpu())
        ens_preds.append(ens_hard.cpu())

        if save_csv is not None:
            for i, title in enumerate(titles):
                rows.append({
                    "title": str(title),
                    "y_true_log10lc50": float(y[i].cpu().item()),
                    "y_pred_log10lc50": float(pred_reg[i].cpu().item()),
                    "y_true_bin": int(yb[i].cpu().item()),
                    "pred_soft_bin": int(soft_hard[i].cpu().item()),
                    "pred_coral_bin": int(coral_hard[i].cpu().item()),
                    "pred_ens_bin":  int(ens_hard[i].cpu().item()),
                })

    # metrics
    rmse = (sse/max(n,1))**0.5
    mean_y = sum_y / max(n,1)
    sst = max(1e-12, sum_y2 - n*(mean_y**2))
    r2 = 1.0 - (sse / sst) if sst > 0 else float('nan')

    y_true = torch.cat(true_bins).numpy()
    y_soft = torch.cat(soft_preds).numpy()
    y_coral = torch.cat(coral_preds).numpy()
    y_ens  = torch.cat(ens_preds).numpy()

    def acc(yhat): return float((yhat == y_true).mean())
    def w1b(yhat): return float((np.abs(yhat - y_true) <= 1).mean())

    acc_soft, acc_coral, acc_ens = acc(y_soft), acc(y_coral), acc(y_ens)
    w1b_soft, w1b_coral, w1b_ens = w1b(y_soft), w1b(y_coral), w1b(y_ens)

    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv, index=False)

    return rmse, r2, acc_soft, acc_coral, acc_ens, w1b_soft, w1b_coral, w1b_ens


class EarlyStopper:
    def __init__(self, mode: str = "max", patience: int = 8, min_delta: float = 0.0, restore_best: bool = True):
        assert mode in ("max", "min")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = -1

    def _is_improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > (self.best + self.min_delta)
        else:
            return value < (self.best - self.min_delta)

    def step(self, value: float, model: nn.Module, optimizer=None, scheduler=None, epoch: int = -1) -> bool:
        if self._is_improved(value):
            self.best = value
            self.num_bad = 0
            if self.restore_best:
                self.best_state = {
                    "epoch": epoch,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optim": optimizer.state_dict() if optimizer is not None else None,
                    "sched": scheduler.state_dict() if scheduler is not None else None,
                }
                self.best_epoch = epoch
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience

    def restore(self, model: nn.Module, optimizer=None, scheduler=None):
        if not self.restore_best or self.best_state is None:
            return
        model.load_state_dict(self.best_state["model"])
        if optimizer is not None and self.best_state["optim"] is not None:
            optimizer.load_state_dict(self.best_state["optim"])
        if scheduler is not None and self.best_state["sched"] is not None:
            scheduler.load_state_dict(self.best_state["sched"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MolNet with simple CORAL + simple tuner')
    parser.add_argument('--train_data', type=str, default='./molnet_dataset/dataset_group_s1_train_flat.pkl')
    parser.add_argument('--test_data', type=str,  default='./molnet_dataset/dataset_group_s1_test_flat.pkl')
    parser.add_argument('--model_config_path', type=str, default='./config/molnet_5.yml')
    parser.add_argument('--checkpoint_path', type=str,   default='./check_point/molnet_coral_simple.pt')
    parser.add_argument('--resume_path', type=str,       default='')
    parser.add_argument('--transfer', action='store_true', default=False)
    parser.add_argument('--ex_model_path', type=str, default='')
    parser.add_argument("--pred_csv_best", type=str, default="./analysis_outputs/pred_3d_best_joint.csv")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--epoch_csv_path', type=str, default='')
    parser.add_argument('--calibrate_at', type=int, default=8, help='Epoch to run the simple F1 tuner once (0=disable)')
    parser.add_argument('--patience', type=int, default=8, help='Early-stop patience on acc_coral')
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.model_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('Load config from', args.model_config_path)

    # Data
    train_set = MolMS_Dataset(args.train_data, data_augmentation=False, conformer_mode='single')
    train_loader = DataLoader(
        train_set,
        batch_size=config['train']['batch_size'],
        sampler=make_balanced_sampler(train_set),
        num_workers=config['train']['num_workers'],
        drop_last=False,
        collate_fn=collate_molms
    )
    valid_set = MolMS_Dataset(args.test_data, data_augmentation=False, conformer_mode='single')
    valid_loader = DataLoader(
        valid_set,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        drop_last=False,
        collate_fn=collate_molms
    )

    # Model/opt
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    model = MolNet_MS(config['model']).to(device)
    base_lr = config['train']['lr']

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)

    if args.resume_path:
        print("Loading checkpoint:", args.resume_path)
        ckpt = torch.load(args.resume_path, map_counter=lambda k: k, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    if args.checkpoint_path:
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    monitor = "acc_coral"
    best_val = -float("inf")
    coral_biases = None
    early = EarlyStopper(mode="max", patience=args.patience, min_delta=1e-4, restore_best=True)
    warmup_epoch    = 3
    calibrate_every = 3
    alpha_temp_grid = None
    for epoch in range(1, int(config['train']['epochs']) + 1):
        print(f"\n=== Epoch {epoch}")

        # Train
        tr_rmse, tr_reg, tr_ce, tr_coral = train_one_epoch(
            model, device, train_loader, optimizer,
            w_reg=0.20, w_ce=0.80, w_coral=0.10
        )
        print(f"Train RMSE {tr_rmse:.4f} | (reg:{tr_reg:.4f}, ce:{tr_ce:.4f}, coral:{tr_coral:.4f})")

        val_logits, val_y = gather_coral_logits_and_targets(model, device, valid_loader)

        if epoch >= warmup_epoch and ((epoch - warmup_epoch) % calibrate_every == 0):
            alpha, thr, f1 = tune_coral_thresholds_f1_simple(
                val_logits, val_y,
                thr_grid=tuple(float(x) for x in np.arange(0.30, 0.71, 0.02)),
                coord_rounds=2,
                temp_grid=alpha_temp_grid
            )
            coral_biases = thresholds_to_biases(thr).to(device)
            print(f"[CAL] macro-F1={f1:.4f}  alpha={alpha:.2f}  thr={thr.cpu().numpy().round(2)}")
        # Eval
        val_rmse, val_r2, acc_s, acc_coral, acc_e, w1b_s, w1b_coral, w1b_e = eval_joint(
            model, device, valid_loader, coral_biases=coral_biases, ensemble=True, save_csv=None
        )
        print(f"Valid RMSE {val_rmse:.4f} R2 {val_r2:.4f} | "
              f"ACC soft {acc_s:.4f} CORAL {acc_coral:.4f} ENS {acc_e:.4f} | "
              f"W1B soft {w1b_s:.4f} CORAL {w1b_coral:.4f} ENS {w1b_e:.4f}  (monitor={monitor})")

        _csv_append_row(
            args.epoch_csv_path,
            {
                "epoch": epoch,
                "train_rmse": float(tr_rmse),
                "train_loss_reg": float(tr_reg),
                "train_loss_ce": float(tr_ce),
                "train_loss_coral": float(tr_coral),
                "val_rmse": float(val_rmse),
                "val_r2": float(val_r2),
                "acc_soft": float(acc_s),
                "acc_coral": float(acc_coral),
                "acc_ens": float(acc_e),
                "w1b_soft": float(w1b_s),
                "w1b_coral": float(w1b_coral),
                "w1b_ens": float(w1b_e),
                "lr": float(get_lr(optimizer)),
                "has_coral_biases": int(coral_biases is not None),
            }
        )

        scheduler.step(acc_coral)

        if acc_coral > best_val:
            best_val = acc_coral
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "tuned_thr": (coral_biases.detach().cpu() if coral_biases is not None else None),
                "best_metric": best_val,
                "monitor": monitor,
                "mode": "max",
            }
            if args.checkpoint_path:
                torch.save(best_state, args.checkpoint_path)
            print(f" New best {monitor}: {best_val:.4f}")

        should_stop = early.step(acc_coral, model, optimizer, scheduler, epoch)
        print(f"[EarlyStop] best {monitor}={early.best:.4f} @ epoch {early.best_epoch} | bad_epochs={early.num_bad}/{early.patience}")
        if should_stop:
            print("Early stopping triggered — restoring best weights.")
            early.restore(model, optimizer, scheduler)
            break

    print(f"\nBest validation {monitor}: {early.best:.4f} (epoch {early.best_epoch})")
    if args.checkpoint_path:
        torch.save({
            "epoch": early.best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tuned_thr": None, 
            "best_metric": early.best,
            "monitor": monitor,
            "mode": "max",
        }, args.checkpoint_path)
    print(f"Checkpoint saved to: {args.checkpoint_path}")
