#!/usr/bin/env python3
# cluster_side_by_side.py 

import os, re, math, hashlib, random
from typing import Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from matplotlib.patches import Ellipse

# Optional packages
try:
    import umap  
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan  
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def wilson_interval(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def numeric_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s))

def detect_fp_cols(df, start_index=20, non_fp_hints=None):
    if start_index is not None and start_index < len(df.columns):
        return list(df.columns[start_index:])
    if non_fp_hints is None: non_fp_hints = set()
    fp_cols = []
    for c in df.columns:
        if c in non_fp_hints: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 1:
                fp_cols.append(c)
        else:
            try:
                sample = pd.to_numeric(df[c].head(100), errors="coerce")
                if sample.notna().sum() > 50: fp_cols.append(c)
            except Exception:
                pass
    return fp_cols

def coerce_numeric_frame(df):
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.notna().any(axis=0)]
    out = out.fillna(0.0)
    nunique = out.nunique(axis=0)
    out = out.loc[:, nunique > 1]
    return out

def is_mostly_binary(X, thresh=0.98):
    d = X.shape[1]
    bin_cols = 0
    for j in range(d):
        u = np.unique(X[:, j])
        if len(u) <= 3 and set(np.round(u).astype(int)).issubset({0, 1}):
            bin_cols += 1
    return (bin_cols / max(d, 1)) >= thresh

def make_clusterfp_kmeans(X_cont, k=64, tau=0.9, random_state=42):

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_cont)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(Xz)
    D = pairwise_distances(Xz, km.cluster_centers_, metric="euclidean")  # [n,k]
    Z = -D / max(1e-6, float(tau))
    Z -= Z.max(axis=1, keepdims=True)
    S = np.exp(Z); S /= S.sum(axis=1, keepdims=True)
    return S, scaler, km

def embed_2d_auto(X, is_binary_fp, seed=42, prefer="tsne", pca_dim=50):

    n, d = X.shape
    Xp = X
    if d > pca_dim:
        n_comp = min(pca_dim, max(2, n-1))
        Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    if prefer == "umap" and HAS_UMAP:
        if is_binary_fp:
            reducer = umap.UMAP(n_neighbors=25, min_dist=0.05, spread=2.5, random_state=seed)
        else:
            reducer = umap.UMAP(n_neighbors=min(30, max(10, n//20)), min_dist=0.15,
                                metric="euclidean", random_state=seed)
        return reducer.fit_transform(Xp)

    perplexity = int(max(5, min(50, (n-1)//3)))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate=100,       
        perplexity=60,           
        n_iter=3000,            
        early_exaggeration=8.0,  
        angle=0.5,
        metric="euclidean" if not is_binary_fp else "cosine",
        random_state=seed,
        verbose=0
    )

    return tsne.fit_transform(Xp)

def centers_from_embedding(Z, labels, mode="medoid"):
    centers = []
    uniq = [u for u in np.unique(labels) if u >= 0]
    for k in uniq:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            centers.append([np.nan, np.nan]); continue
        pts = Z[idx]
        if mode == "mean":
            centers.append(pts.mean(axis=0))
        else:
            mu = pts.mean(axis=0)
            j = np.argmin(np.sum((pts - mu) ** 2, axis=1))
            centers.append(pts[j])
    return np.array(centers)

def signature(ids, k, tau, seed):
    m = hashlib.sha1()
    m.update(("|".join(ids)).encode())
    m.update(f"|k={k}|tau={tau}|seed={seed}".encode())
    return m.hexdigest()[:12]

def build_alignment_ids(
    pred: pd.DataFrame,
    test_fp: pd.DataFrame,
    pred_key: str,
    test_key: str,
    id_map_csv: Optional[str] = None,
):

    join_col = pred_key
    if id_map_csv is not None:
        m = pd.read_csv(id_map_csv)
        if pred_key not in m.columns or test_key not in m.columns:
            raise KeyError(f"id_map_csv must have columns '{pred_key}' and '{test_key}'")
        m[pred_key] = m[pred_key].astype(str)
        m[test_key] = m[test_key].astype(str)
        pred = pred.merge(m[[pred_key, test_key]].drop_duplicates(), on=pred_key, how="left")
        join_col = test_key 

    pred_ids = pred[join_col].astype(str)
    test_ids = test_fp[test_key].astype(str)
    overlap = set(pred_ids) & set(test_ids)

    if len(overlap) > 0:
        fp_cols = [c for c in test_fp.columns if c != test_key]
        raw_fp = (test_fp[[test_key] + fp_cols]
                  .drop_duplicates(test_key)
                  .set_index(test_key))
        fp_numeric = coerce_numeric_frame(raw_fp)
        ids_series = pred_ids
        ids_series = ids_series[ids_series.isin(fp_numeric.index.astype(str))]
        return ids_series, fp_numeric

    if pred_key == "title":
        pred = pred.copy()
        pred["_pred_numid"] = pred[pred_key].astype(str).str.extract(r"^(\d+)", expand=False).fillna("")
        test_fp = test_fp.copy()
        test_fp["_test_numid"] = test_fp[test_key].astype(str).map(numeric_only)
        if (pred["_pred_numid"] != "").sum() == 0:
            raise ValueError("Could not extract numeric prefix from any 'title'. Provide id_map_csv.")

        pred_ids = pred["_pred_numid"]
        test_ids = test_fp["_test_numid"]
        overlap = set(pred_ids) & set(test_ids)
        if len(overlap) == 0:
            sample_titles = pred[pred_key].head(5).tolist()
            raise ValueError(
                "No overlap after numeric-prefix mapping.\n"
                f"Example titles: {sample_titles}\n"
                f"Consider supplying id_map_csv with columns [{pred_key}, {test_key}]"
            )

        fp_cols = [c for c in test_fp.columns if c not in {test_key, "_test_numid"}]
        raw_fp = (test_fp[["_test_numid"] + fp_cols]
                  .drop_duplicates("_test_numid")
                  .set_index("_test_numid"))
        fp_numeric = coerce_numeric_frame(raw_fp)
        ids_series = pred_ids
        ids_series = ids_series[ids_series.isin(fp_numeric.index.astype(str))]
        return ids_series, fp_numeric

    raise ValueError(
        "No direct ID overlap. If keys differ, provide id_map_csv with columns "
        f"[{pred_key}, {test_key}]"
    )

def run_cluster_side_by_side(
    pred2d_csv,       
    pred3d_csv,              
    test_withfp_csv,        
    compound_col="CAS",   
    pred3d_key="title",        
    test_key="CAS Number",    
    id_map_csv: Optional[str] = None, 
    color_by="w1b",             
    fp_start_index=20,
    k_clusterfp=64,
    tau_clusterfp=1.0,
    prefer_embedder="tsne",     
    seed=42,
    out_dir="./analysis_outputs",
    show=True,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.manifold import TSNE

    os.makedirs(out_dir, exist_ok=True)

    def _cov_ellipse(xy: np.ndarray, n_std: float = 2.0):
        if xy.shape[0] < 2:
            return 0.0, 0.0, 0.0
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        return float(w), float(h), float(ang)

    def highlight_clusters_on_ax(ax, df_xy, selected_csv, draw_ellipse=True, n_std=2.0):
        sel = pd.read_csv(selected_csv)
        if "cluster" not in sel.columns or "direction" not in sel.columns:
            raise ValueError("selected_csv must contain columns: 'cluster', 'direction'")
        sel["cluster"] = sel["cluster"].astype(int)
        set_3d = set(sel.loc[sel["direction"] == "3D>>2D", "cluster"])
        set_2d = set(sel.loc[sel["direction"] == "2D>>3D", "cluster"])

        def _draw(cluster_ids, fc, ec, label):
            if not cluster_ids:
                return
            sub = df_xy[df_xy["cluster"].isin(cluster_ids)]
            ax.scatter(sub["x"], sub["y"], s=42, facecolors=fc, edgecolors=ec,
                       linewidths=1.2, alpha=0.95, label=label)
            if draw_ellipse and len(sub) >= 3:
                pts = sub[["x", "y"]].to_numpy()
                w, h, ang = _cov_ellipse(pts, n_std=n_std)
                cx, cy = pts.mean(axis=0)
                ax.add_patch(Ellipse((cx, cy), w, h, angle=ang,
                                     fill=False, lw=2.0, ec=ec, alpha=0.9))
            for cid, grp in sub.groupby("cluster"):
                cx, cy = grp[["x", "y"]].mean()
                ax.text(cx, cy, f"{cid}", fontsize=10, weight="bold",
                        ha="center", va="center", color=ec,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0))

        _draw(set_3d, (0.10, 0.55, 0.95, 0.25), "#1677ff", "3D >> 2D")
        _draw(set_2d, (0.95, 0.35, 0.10, 0.25), "#ff6a13", "2D >> 3D")
        ax.legend(frameon=True, loc="best")

    try:
        embed_2d_auto  # type: ignore
    except NameError:
        def embed_2d_auto(X, is_binary_fp=False, seed=seed, prefer="tsne"):
            return TSNE(n_components=2, random_state=seed, init="random", learning_rate="auto").fit_transform(X)

    pred2d = pd.read_csv(pred2d_csv)
    pred3d = pd.read_csv(pred3d_csv)
    test_fp = pd.read_csv(test_withfp_csv)

    if compound_col not in pred2d.columns or compound_col not in test_fp.columns:
        raise KeyError(f"'{compound_col}' must be present in both pred2d and test files.")
    pred2d[compound_col] = pred2d[compound_col].astype(str)
    test_fp[test_key] = test_fp[test_key].astype(str)  # for panel B mapping

    corr2d_col = "correct_w1b" if (color_by.lower() == "w1b" and "correct_w1b" in pred2d.columns) else "correct_exact"
    if corr2d_col not in pred2d.columns:
        raise KeyError("pred2d_csv must contain 'correct_w1b' or 'correct_exact'.")

    agg2d = (pred2d.groupby(compound_col, as_index=False)
             .agg(n_obs=(corr2d_col, "size"),
                  n_correct=(corr2d_col, "sum")))
    agg2d["p_correct"] = agg2d["n_correct"] / agg2d["n_obs"]
    lo2, hi2 = [], []
    for k, n in zip(agg2d["n_correct"].values, agg2d["n_obs"].values):
        l, h = wilson_interval(int(k), int(n)); lo2.append(l); hi2.append(h)
    agg2d["p_lo"], agg2d["p_hi"] = lo2, hi2

    if "y_true_bin" not in pred3d.columns or "pred_coral_bin" not in pred3d.columns:
        raise KeyError("pred3d_csv must contain 'y_true_bin' and 'pred_coral_bin'.")
    pred3d[pred3d_key] = pred3d[pred3d_key].astype(str)
    pred3d = pred3d.copy()
    pred3d["correct_exact"] = (pred3d["y_true_bin"].astype(int) == pred3d["pred_coral_bin"].astype(int)).astype(int)
    pred3d["correct_w1b"]   = (np.abs(pred3d["y_true_bin"].astype(int) - pred3d["pred_coral_bin"].astype(int)) <= 1).astype(int)
    corr3d_col = "correct_w1b" if color_by.lower() == "w1b" else "correct_exact"

    ids_series, fp_numeric = build_alignment_ids(pred3d, test_fp, pred3d_key, test_key, id_map_csv=id_map_csv)

    pred3d_aligned = pred3d.copy()
    pred3d_aligned["_align_key"] = ids_series.values
    pred3d_aligned = pred3d_aligned[pred3d_aligned["_align_key"].notna()]

    agg3d = (pred3d_aligned.groupby("_align_key", as_index=False)
             .agg(n_obs=(corr3d_col, "size"),
                  n_correct=(corr3d_col, "sum")))
    agg3d["p_correct"] = agg3d["n_correct"] / agg3d["n_obs"]
    lo3, hi3 = [], []
    for k, n in zip(agg3d["n_correct"].values, agg3d["n_obs"].values):
        l, h = wilson_interval(int(k), int(n)); lo3.append(l); hi3.append(h)
    agg3d["p_lo"], agg3d["p_hi"] = lo3, hi3

    ids_union = list(dict.fromkeys(
        list(agg2d[compound_col].astype(str).values) +
        list(agg3d["_align_key"].astype(str).values)
    ))

    non_fp_hints = {compound_col, test_key, "Effect value", "Duration (hours)",
                    "y_true_bin", "y_pred_bin", "y_true_log10lc50", "y_pred_log10lc50",
                    "correct_w1b", "correct_exact", "Latin name", "_test_numid"}
    fp_cols = detect_fp_cols(test_fp, start_index=fp_start_index, non_fp_hints=non_fp_hints)
    if not fp_cols:
        raise ValueError("No fingerprint columns detected; check fp_start_index or dataset schema.")

    key_for_fp = compound_col if compound_col in test_fp.columns else test_key
    fps_indexed = (test_fp[[key_for_fp] + fp_cols]
                   .drop_duplicates(subset=[key_for_fp], keep="first")
                   .set_index(key_for_fp).astype(float))
    fps_numeric_full = coerce_numeric_frame(fps_indexed)

    ids_exist = [i for i in ids_union if i in fps_numeric_full.index]
    if not ids_exist:
        raise ValueError("After alignment, 0 IDs found in test fingerprints. Check keys/mapping.")
    X = fps_numeric_full.loc[ids_exist].values.astype(np.float64, copy=False)

    S, scaler, km = make_clusterfp_kmeans(X, k=k_clusterfp, tau=tau_clusterfp, random_state=seed)
    labels = km.predict(scaler.transform(X))  
    cluster_map = pd.Series(labels, index=ids_exist).to_dict()

    S_logit = np.log(np.clip(S, 1e-6, 1-1e-6) / np.clip(1 - S, 1e-6, 1-1e-6))
    sig = signature(ids_exist, k_clusterfp, tau_clusterfp, seed)
    emb_cache = os.path.join(out_dir, f"embedding_{sig}.csv")
    if os.path.exists(emb_cache):
        Z = pd.read_csv(emb_cache).values
        if Z.shape[0] != len(ids_exist):
            Z = embed_2d_auto(S_logit, is_binary_fp=False, seed=seed, prefer=prefer_embedder)
            pd.DataFrame(Z, columns=["x", "y"]).to_csv(emb_cache, index=False)
    else:
        Z = embed_2d_auto(S_logit, is_binary_fp=False, seed=seed, prefer=prefer_embedder)
        pd.DataFrame(Z, columns=["x", "y"]).to_csv(emb_cache, index=False)

    coords = pd.DataFrame({"id": ids_exist, "x": Z[:, 0], "y": Z[:, 1]})
    coords["cluster"] = coords["id"].map(cluster_map).astype(int)
    coords = coords.set_index("id")

    A = agg2d.merge(coords, left_on=compound_col, right_index=True, how="inner")
    B = agg3d.merge(coords, left_on="_align_key", right_index=True, how="inner")

    plt.figure(figsize=(14, 6))
    vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap("RdYlGn")

    allx = np.concatenate([A["x"].values, B["x"].values])
    ally = np.concatenate([A["y"].values, B["y"].values])
    pad = 2.0
    xlim = (allx.min() - pad, allx.max() + pad)
    ylim = (ally.min() - pad, ally.max() + pad)

    ax1 = plt.subplot(1, 2, 1)
    sizesA = 30 + 120 * (np.sqrt(A["n_obs"]) / np.sqrt(max(A["n_obs"].max(), 1)))
    sc1 = ax1.scatter(A["x"], A["y"], c=A["p_correct"], cmap=cmap, vmin=vmin, vmax=vmax,
                      s=sizesA, alpha=0.95, linewidths=0.4, edgecolors="k")
    ax1.set_title(f"2D GPBoost correctness ({'W1B' if color_by.lower()=='w1b' else 'exact'})")
    ax1.set_xlabel("Embedding-1"); ax1.set_ylabel("Embedding-2")
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)

    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    sizesB = 30 + 120 * (np.sqrt(B["n_obs"]) / np.sqrt(max(B["n_obs"].max(), 1)))
    sc2 = ax2.scatter(B["x"], B["y"], c=B["p_correct"], cmap=cmap, vmin=vmin, vmax=vmax,
                      s=sizesB, alpha=0.95, linewidths=0.4, edgecolors="k")
    ax2.set_title(f"3D Mol Tox correctness ({'W1B' if color_by.lower()=='w1b' else 'exact'})")
    ax2.set_xlabel("Embedding-1"); ax2.set_ylabel("")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc2, cax=cax)
    cbar.set_label("Correctness rate", rotation=270, labelpad=15)

    baseA = os.path.splitext(os.path.basename(pred2d_csv))[0]
    baseB = os.path.splitext(os.path.basename(pred3d_csv))[0]
    selected_csv = os.path.join(out_dir, "cluster_stats",
                                f"cluster_selected__{baseA}__{baseB}.csv")
    # if os.path.exists(selected_csv):
    #     highlight_clusters_on_ax(ax1, A[["x", "y", "cluster"]], selected_csv, draw_ellipse=False, n_std=2.0)
    #     highlight_clusters_on_ax(ax2, B[["x", "y", "cluster"]], selected_csv, draw_ellipse=False, n_std=2.0)
    # else:
    #     print(f"[WARN] Selected clusters file not found: {selected_csv}")

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"side_by_side__{baseA}__{baseB}.png")
    plt.savefig(out_png, dpi=300)
    plt.show() if show else plt.close()

    A_out = os.path.join(out_dir, f"{baseA}__coords.csv")
    B_out = os.path.join(out_dir, f"{baseB}__coords.csv")
    A.to_csv(A_out, index=False); B.to_csv(B_out, index=False)
    print(f"[OK] Saved figure: {out_png}")
    print(f"[OK] Saved panel tables:\n  {A_out}\n  {B_out}")



def main():
    pred2d_csv = "./analysis_outputs/scaffold/pred__dataset__mordred__scaffold__s3__na__classification__GPBoost.csv"
    pred3d_csv = "./analysis_outputs/pred_3d_best_coral.csv"
    test_withfp_csv = "F:/molnet_dataset_nodot_flat/dataset_scaffold_s3_test_flat.csv"

    run_cluster_side_by_side(
        pred2d_csv=pred2d_csv,
        pred3d_csv=pred3d_csv,
        test_withfp_csv=test_withfp_csv,
        compound_col="CAS",         
        pred3d_key="title",        
        test_key="CAS",   
        id_map_csv=None,           
        color_by="w1b",            
        fp_start_index=20,
        k_clusterfp=64,
        tau_clusterfp=1.0,
        prefer_embedder="tsne",     
        seed=SEED,
        out_dir="./analysis_outputs",
        show=True,
    )

if __name__ == "__main__":
    main()
