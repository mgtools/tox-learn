#!/usr/bin/env python3
# clusterfp_plot.py  

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

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


def wilson_interval(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def detect_fp_cols(df, start_index=20, non_fp_hints=None):
    if start_index is not None and start_index < len(df.columns):
        return list(df.columns[start_index:])
    if non_fp_hints is None: non_fp_hints = set()
    fp_cols = []
    for c in df.columns:
        if c in non_fp_hints: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 10: 
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

def embed_2d_auto(X, is_binary_fp, seed=42, prefer="umap", pca_dim=50):

    n, d = X.shape
    Xp = X
    if d > pca_dim:
        n_comp = min(pca_dim, max(2, n-1))
        Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    if prefer == "umap" and HAS_UMAP:
        if is_binary_fp:
            reducer = umap.UMAP(
                n_neighbors=25,       
                min_dist=0.05,        
                spread=2.5,          
                random_state=seed
            )
        else:
            reducer = umap.UMAP(
                n_neighbors=min(30, max(10, n//20)),
                min_dist=0.15,
                metric="euclidean",
                random_state=seed
            )
        return reducer.fit_transform(Xp)

    # t-SNE
    perplexity = int(max(5, min(50, (n-1)//3)))
    metric = "cosine" if is_binary_fp else "euclidean"
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate=50,     
        perplexity=40,           
        n_iter=2500,             
        early_exaggeration=16.0,  
        angle=0.4,
        metric="euclidean",
        random_state=seed,
        verbose=1
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


def run_clusterfp_plot(
    pred_csv,
    test_withfp_csv,
    compound_col="CAS",
    use_w1b=True,
    fp_start_index=20,     
    k_clusterfp=64,      
    tau_clusterfp=0.9,   
    use_hdbscan=True,   
    kmeans_on_2d=10,  
    seed=42,
    out_dir="./analysis_outputs",
    show=True,
):
    os.makedirs(out_dir, exist_ok=True)

    # load
    pred = pd.read_csv(pred_csv)
    test_fp = pd.read_csv(test_withfp_csv)
    if compound_col not in pred.columns or compound_col not in test_fp.columns:
        raise ValueError(f"Join column '{compound_col}' must exist in both files.")
    pred[compound_col] = pred[compound_col].astype(str)
    test_fp[compound_col] = test_fp[compound_col].astype(str)

    # correctness aggregation
    corr_col = "correct_w1b" if (use_w1b and "correct_w1b" in pred.columns) else "correct_exact"
    if corr_col not in pred.columns:
        raise KeyError("Prediction CSV must contain either 'correct_w1b' or 'correct_exact'.")

    agg = (pred.groupby(compound_col, as_index=False)
                .agg(n_obs=(corr_col, "size"),
                     n_correct=(corr_col, "sum")))
    agg["p_correct"] = agg["n_correct"] / agg["n_obs"]
    ci = agg.apply(lambda r: wilson_interval(r["n_correct"], r["n_obs"]), axis=1, result_type="expand")
    agg["p_lo"], agg["p_hi"] = ci[0], ci[1]

    # pick FP columns and make numeric
    non_fp_hints = {
        compound_col, "Latin name", "Effect value", "Duration (hours)",
        "y_true_bin", "y_pred_bin", "y_true_log10lc50", "y_pred_log10lc50",
        "correct_w1b", "correct_exact"
    }
    fp_cols = detect_fp_cols(test_fp, start_index=fp_start_index, non_fp_hints=non_fp_hints)
    raw_fp = (test_fp[[compound_col] + fp_cols]
              .drop_duplicates(subset=[compound_col], keep="first")
              .set_index(compound_col))
    fp_numeric = coerce_numeric_frame(raw_fp)

    # align IDs
    ids = agg[compound_col].astype(str)
    missing = set(ids) - set(fp_numeric.index)
    if missing:
        agg = agg[~agg[compound_col].isin(missing)].reset_index(drop=True)
        ids = agg[compound_col].astype(str)

    X = fp_numeric.loc[ids].values.astype(np.float64, copy=False)
    binary_fp = is_mostly_binary(X)

    # representation for embedding
    if binary_fp:
        X_embed = X
        proto_labels = None
        rep_tag = "bitvector"
    else:
        X_embed, _, _ = make_clusterfp_kmeans(X, k=k_clusterfp, tau=tau_clusterfp, random_state=seed)
        proto_labels = X_embed.argmax(axis=1)  
        rep_tag = f"ClusterFP(K={k_clusterfp},tau={tau_clusterfp})"

    S_logit = np.log(np.clip(X_embed, 1e-6, 1-1e-6) / np.clip(1 - X_embed, 1e-6, 1-1e-6))
    Z = embed_2d_auto(S_logit, is_binary_fp=False, seed=seed, prefer="tsne")
    agg["x"], agg["y"] = Z[:, 0], Z[:, 1]

    if HAS_HDBSCAN and use_hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(10, len(agg)//50),
                                    min_samples=None, metric='euclidean')
        labels = clusterer.fit_predict(Z)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[HDBSCAN] clusters found: {n_found} (label -1=noise)")
    else:
        k = max(2, int(kmeans_on_2d))
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)

    agg["cluster"] = labels

    lbl = labels.copy()
    mask_valid = lbl >= 0
    centers_2d = centers_from_embedding(Z[mask_valid], lbl[mask_valid]) if np.any(mask_valid) else None

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("RdYlGn")
    sizes = 40 + 120 * (np.sqrt(agg["n_obs"]) / np.sqrt(max(agg["n_obs"].max(), 1)))
    sc = plt.scatter(agg["x"], agg["y"],
                     c=agg["p_correct"].values, cmap=cmap, vmin=0.0, vmax=1.0,
                     s=sizes, alpha=0.95, linewidths=0.4, edgecolors="k", label="compound")

    mask_ci = (agg["p_lo"] <= 0.5) & (agg["p_hi"] >= 0.5)
    if mask_ci.any():
        plt.scatter(agg.loc[mask_ci, "x"], agg.loc[mask_ci, "y"],
                    facecolors="none", edgecolors="black", linewidths=1.2,
                    s=(sizes[mask_ci.values] * 1.15), label="CI crosses 0.5")

    if centers_2d is not None and centers_2d.size:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                    marker="X", s=160, c="none", edgecolors="k", linewidths=1.2,
                    label="cluster center")

    cbar = plt.colorbar(sc, shrink=0.85, pad=0.02)
    cbar.set_label(f"Correctness rate ({'W1B' if use_w1b else 'exact'})")

    em_tag = "UMAP" if HAS_UMAP else "t-SNE"
    ttl = f"Compound embedding; color=correctness"
    plt.title(ttl)
    plt.xlabel("Embedding-1"); plt.ylabel("Embedding-2")
    plt.legend(loc="best", frameon=True); plt.tight_layout()

    base = os.path.splitext(os.path.basename(pred_csv))[0]
    out_png = os.path.join(out_dir, f"{base}__clusterfp_plot.png")
    out_csv = os.path.join(out_dir, f"{base}__clusterfp_embedding.csv")
    plt.savefig(out_png, dpi=300)
    plt.show() if show else plt.close()

    agg.to_csv(out_csv, index=False)
    print(f"[INFO] Saved figure: {out_png}")
    print(f"[INFO] Saved embedding table: {out_csv}")


def main():
    pred_csv = "./analysis_outputs/scaffold/pred__dataset__mordred__scaffold__s3__na__classification__GPBoost.csv"
    test_withfp_csv = "F:/molnet_dataset_nodot_flat/dataset_scaffold_s3_test_flat.csv"

    run_clusterfp_plot(
        pred_csv=pred_csv,
        test_withfp_csv=test_withfp_csv,
        compound_col="CAS",
        use_w1b=True,
        fp_start_index=20,     
        k_clusterfp=64,
        tau_clusterfp=1.2,
        use_hdbscan=True,
        kmeans_on_2d=10,
        seed=42,
        out_dir="./analysis_outputs",
        show=True,
    )

if __name__ == "__main__":
    main()
