"""
Trains and saves the four artefacts the CORE dashboard loads at startup:

  scaler.pkl        StandardScaler fitted on the 250 000 fault-free samples
  pca_model.pkl     PCA(31) + T2/Q control limits + loadings/eigenvalues
  lgbm_model.pkl    LightGBM binary classifier (Normal = 0 vs Fault = 1)
  feature_cols.txt  the 52 process variables, in the order the models expect

The faulty training file is 1.9 GB, so it is read in chunks and sampled with a
fixed stride, only ONE CHUNK is in RAM at a time, same approach as
prep_demo_data.py.

Usage:
    python save_models.py                 # write artefacts into the project root
    python save_models.py --out-dir tmp   # write elsewhere (e.g. to compare runs)

Reproducibility
---------------
Re-running this reproduces the shipped MSPC artefacts essentially exactly: the
scaler's mean_/scale_ and the PCA eigenvalues come back bit-identical, T2_UCL
matches to the last digit, and Q_UCL to within 7e-8 relative.

The LightGBM model is retrained, not reproduced bit-for-bit. The original run's
train/validation split was not recorded, so early stopping lands on a different
round (the shipped model stopped at 372 of 500). Hyper-parameters, feature
handling and input dtype are identical.
"""

import argparse, os, pickle, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))

FAULTFREE_TRAIN = os.path.join(BASE, "TEP_FaultFree_Training.csv")   # 250 000 rows
FAULTY_TRAIN    = os.path.join(BASE, "TEP_Faulty_Training.csv")      # 5 000 000 rows

# Columns that describe the run rather than the process itself.
META_COLS = ["faultNumber", "simulationRun", "sample"]

N_COMPONENTS = 31      # 90.1% of variance on the fault-free baseline
CONFIDENCE   = 0.99    # control limits at 99%
N_FAULTY     = 250_000 # balanced against the 250 000 fault-free rows
SEED         = 42


# ── control limits ────────────────────────────────────────────────────────────
def t2_limit(n_samples, n_components, confidence=CONFIDENCE):
    """Hotelling's T² upper control limit (F-distribution form)."""
    a, n = n_components, n_samples
    return a * (n**2 - 1) / (n * (n - a)) * stats.f.ppf(confidence, a, n - a)


def q_limit(discarded_eigenvalues, confidence=CONFIDENCE):
    """Q (SPE) upper control limit, Jackson & Mudholkar.

    Built from the eigenvalues PCA threw away, so it needs the full-rank
    spectrum, not just the retained components.
    """
    d = np.asarray(discarded_eigenvalues, dtype=np.float64)
    th1, th2, th3 = d.sum(), (d**2).sum(), (d**3).sum()
    h0 = 1 - (2 * th1 * th3) / (3 * th2**2)
    ca = stats.norm.ppf(confidence)
    return th1 * ((ca * np.sqrt(2 * th2 * h0**2) / th1)
                  + 1
                  + (th2 * h0 * (h0 - 1) / th1**2)) ** (1 / h0)


# ── data loading ──────────────────────────────────────────────────────────────
def load_faultfree():
    """The full fault-free training file, 250 000 rows, ~50 MB as float32."""
    print(f"Reading {os.path.basename(FAULTFREE_TRAIN)} ...")
    df = pd.read_csv(FAULTFREE_TRAIN)
    feat = [c for c in df.columns if c not in META_COLS]
    X = df[feat].to_numpy(dtype=np.float32)
    print(f"  {len(X):,} rows  x  {len(feat)} variables")
    return X, feat


def sample_faulty(feat, n_target=N_FAULTY):
    """Stride-sample the 1.9 GB faulty file without loading it whole.

    A fixed stride is used rather than a random sample so the selection is
    reproducible and spreads evenly across all 20 faults and every simulation
    run, which a head/tail slice would not.
    """
    total_rows = 5_000_000
    stride = total_rows // n_target          # 20 -> every 20th row
    print(f"\nStreaming {os.path.basename(FAULTY_TRAIN)} "
          f"({os.path.getsize(FAULTY_TRAIN)/1e9:.1f} GB), taking every {stride}th row ...")

    parts, seen = [], 0
    for chunk in pd.read_csv(FAULTY_TRAIN, usecols=feat, chunksize=200_000):
        # offset keeps the stride continuous across chunk boundaries
        first = (-seen) % stride
        parts.append(chunk.to_numpy(dtype=np.float32)[first::stride])
        seen += len(chunk)
        pct = seen / total_rows
        print(f"\r  [{'#' * int(pct * 40):<40}] {pct:5.1%}  {seen:>9,} / {total_rows:,} rows",
              end="", flush=True)

    X = np.concatenate(parts)[:n_target]
    print(f"\n  sampled {len(X):,} faulty rows")
    return X


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train and save the CORE detection artefacts")
    ap.add_argument("--out-dir", default=BASE,
                    help="where to write the artefacts (default: project root)")
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)

    for path in (FAULTFREE_TRAIN, FAULTY_TRAIN):
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing {os.path.basename(path)}.\n"
                "Download the TEP dataset first, see the README, Getting Started step 3."
            )

    t_start = time.time()

    # ── 1. scaler, fitted on normal operation only ────────────────────────────
    X_normal, feat = load_faultfree()
    scaler = StandardScaler().fit(X_normal)
    Xs_normal = scaler.transform(X_normal)

    # ── 2. PCA + MSPC control limits ──────────────────────────────────────────
    print(f"\nFitting PCA ({N_COMPONENTS} components) ...")
    t0 = time.time()
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Xs_normal)
    pca_secs = time.time() - t0
    cumvar = pca.explained_variance_ratio_.sum()
    print(f"  {N_COMPONENTS} components explain {cumvar:.1%} of variance  ({pca_secs:.1f} s)")

    # Q's limit depends on the DISCARDED eigenvalues, so fit the full spectrum too.
    full_eigenvalues = PCA(random_state=SEED).fit(Xs_normal).explained_variance_
    T2_UCL = t2_limit(len(Xs_normal), N_COMPONENTS)
    Q_UCL  = q_limit(full_eigenvalues[N_COMPONENTS:])
    print(f"  T2 UCL = {T2_UCL:.4f}    Q UCL = {Q_UCL:.4f}   (at {CONFIDENCE:.0%} confidence)")

    # ── 3. balanced training set ──────────────────────────────────────────────
    X_faulty = sample_faulty(feat)
    X = np.vstack([X_normal, X_faulty])
    y = np.concatenate([np.zeros(len(X_normal), dtype=np.int8),
                        np.ones(len(X_faulty),  dtype=np.int8)])
    del X_normal, X_faulty
    print(f"\nTraining set: {len(X):,} samples "
          f"({(y == 0).sum():,} normal + {(y == 1).sum():,} faulty)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        scaler.transform(X), y, test_size=0.2, random_state=SEED, stratify=y
    )
    del X

    # ── 4. LightGBM ───────────────────────────────────────────────────────────
    print("\nTraining LightGBM ...")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    train_secs = time.time() - t0
    print(f"  stopped at {model.best_iteration_} rounds  ({train_secs:.1f} s)")
    with warnings.catch_warnings():
        # Fitted on arrays, so LightGBM recorded generic Column_N feature names;
        # scoring with a bare array then trips a harmless sklearn name check.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        print(f"  validation accuracy: {model.score(X_val, y_val):.4f}")

    # ── 5. save ───────────────────────────────────────────────────────────────
    with open(os.path.join(out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out, "pca_model.pkl"), "wb") as f:
        pickle.dump({
            "pca":          pca,
            "n_components": N_COMPONENTS,
            "T2_UCL":       T2_UCL,
            "Q_UCL":        Q_UCL,
            "feature_cols": feat,
            "loadings":     pca.components_.T,
            "eigenvalues":  pca.explained_variance_,
        }, f)
    with open(os.path.join(out, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feat))

    print(f"\nSaved to {out}:")
    for name in ("scaler.pkl", "pca_model.pkl", "lgbm_model.pkl", "feature_cols.txt"):
        kb = os.path.getsize(os.path.join(out, name)) / 1024
        print(f"  {name:<20} {kb:>9,.0f} KB")
    print(f"\nDone in {time.time() - t_start:.0f} s.")


if __name__ == "__main__":
    main()
