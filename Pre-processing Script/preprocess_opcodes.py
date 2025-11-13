import os, re, glob, time, sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# file parameters, pulling from code filepath 
DATA_DIR = os.path.abspath("opcodes")
EXPORT_DIR = os.path.abspath("processed_dfs")
os.makedirs(EXPORT_DIR, exist_ok=True)

TEST_SIZE = 0.5
SEED = 42
TFIDF_MAX_FEATURES = 10000 # Cap vocab
TFIDF_MIN_DF = 3
TOKEN_PATTERN = r"\S+" # only non whitespace with x number of characters
LOWERCASE = True
# regex file name extractor, match APT_ strings then pull details between APT_ and the following _'
APT_NAME_RE = re.compile(r'^APT_([^_]+)_', re.IGNORECASE) # regex example re compile ignore case

# KNN rubric (k=3.5) - round to a valid integer
KNN_K = max(1, int(round(3.5)))  # -> 4

def parse_meta(fname: str):
    base = os.path.basename(fname)
    base_nosuf = base[:-7] if base.lower().endswith(".opcode") else base
    m = APT_NAME_RE.match(base_nosuf)
    if m:
        return m.group(1).strip()
    return "UNKNOWN_APT"

def get_filetype(fname: str) -> str:
    base = os.path.basename(fname)
    if base.lower().endswith(".opcode"):
        base = base[:-7]  # drop ".opcode"
    if "." in base:
        return base.rsplit(".", 1)[-1].lower()
    return "unknown"

def read_opcodes(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ops = [ln.strip().lower() for ln in f if ln.strip()]
    return ops

def file_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:,.2f} PB"

def estimate_dense_matrix_mem(n_samples, n_features, dtype_bytes=8):
    # float64 array memory
    return n_samples * n_features * dtype_bytes

def compute_ngram_stats(docs, ngram_range, export_name, export_dir):
    """Compute global n-gram frequencies and save to CSV."""
    print(f"\n[FEATURE CHECK] Computing global {ngram_range[0]}-gram counts...")

    vectorizer = CountVectorizer(ngram_range=ngram_range, token_pattern=TOKEN_PATTERN, lowercase=LOWERCASE)
    term_matrix = vectorizer.fit_transform(docs)
    
    opcode_action = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(term_matrix.sum(axis=0)).ravel()

    df_ngram = pd.DataFrame({"token": opcode_action, "count": counts})
    df_ngram.sort_values("count", ascending=False).to_csv(
        os.path.join(export_dir, f"{export_name}_global_counts.csv"), index=False
    )
    
    print(f"[SAVE] {export_name}_global_counts.csv -> {export_dir} (opcode_action size={len(opcode_action)})")

# BUILD ML DATAFRAMES (1 gram & 2 gram)
def build_ml_dataframe(df_in: pd.DataFrame, ngram=1):
    df_local = df_in.copy()
    df_local["unique_ratio"] = (df_local["unique_opcodes"] / df_local["opcode_count"].replace(0, np.nan)).fillna(0.0)

    # Split BEFORE fitting TF-IDF to avoid leakage
    y_full = df_local["apt"].astype(str)
    tr_idx, te_idx = train_test_split(
        df_local.index, test_size=TEST_SIZE, random_state=SEED, stratify=y_full
    )
    df_tr = df_local.loc[tr_idx].copy()
    df_te = df_local.loc[te_idx].copy()

    print(f"\n[BUILD] {ngram}-gram TF-IDF: train={df_tr.shape[0]} test={df_te.shape[0]}")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram, ngram),
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES
    )
    t0 = time.time()
    Xtext_tr = tfidf.fit_transform(df_tr["doc"])
    Xtext_te = tfidf.transform(df_te["doc"])
    t1 = time.time()
    print(f"[BUILD] TF-IDF fit+transform took {t1 - t0:.2f}s | vocab={len(tfidf.get_feature_names_out())}")

    # DENSE blocks for all models
    text_cols = [f"{ngram}g::{t}" for t in tfidf.get_feature_names_out()]
    Xtext_tr_df = pd.DataFrame(Xtext_tr.toarray(), index=df_tr.index, columns=text_cols)
    Xtext_te_df = pd.DataFrame(Xtext_te.toarray(), index=df_te.index, columns=text_cols)

    # Numeric metadata
    num_cols = ["size_bytes", "opcode_count", "unique_opcodes", "unique_ratio"]
    Xnum_tr_df = df_tr[num_cols].reset_index(drop=True)
    Xnum_te_df = df_te[num_cols].reset_index(drop=True)

    # One-hot filetype (pandas)
    ft_tr = pd.get_dummies(df_tr["filetype"].astype(str).fillna("unknown"), prefix="filetype", drop_first=False)
    ft_te = pd.get_dummies(df_te["filetype"].astype(str).fillna("unknown"), prefix="filetype", drop_first=False)
    ft_te = ft_te.reindex(columns=ft_tr.columns, fill_value=0)

    # Concatenate
    X_train_df = pd.concat([Xtext_tr_df.reset_index(drop=True), Xnum_tr_df, ft_tr.reset_index(drop=True)], axis=1)
    X_test_df  = pd.concat([Xtext_te_df.reset_index(drop=True),  Xnum_te_df,  ft_te.reset_index(drop=True)], axis=1)

    # Debug memory
    mem_train = estimate_dense_matrix_mem(X_train_df.shape[0], X_train_df.shape[1], dtype_bytes=8)
    mem_test  = estimate_dense_matrix_mem(X_test_df.shape[0],  X_test_df.shape[1],  dtype_bytes=8)
    print(f"[BUILD] {ngram}g X_train_df shape={X_train_df.shape} (~{file_bytes(mem_train)})")
    print(f"[BUILD] {ngram}g X_test_df  shape={X_test_df.shape}  (~{file_bytes(mem_test)})")

    # Targets
    y_train = df_tr["apt"].astype(str).reset_index(drop=True)
    y_test  = df_te["apt"].astype(str).reset_index(drop=True)

    # Save artifacts
    fmap = pd.DataFrame({
        "block": (["text"] * len(text_cols)) + (["num"] * len(num_cols)) + (["cat"] * ft_tr.shape[1]),
        "name":  text_cols + num_cols + list(ft_tr.columns)
    })
    fmap.to_csv(os.path.join(EXPORT_DIR, f"feature_map_{ngram}g.csv"), index=False)
    X_train_df.to_csv(os.path.join(EXPORT_DIR, f"X_train_{ngram}g.csv"), index=False)
    X_test_df.to_csv(os.path.join(EXPORT_DIR, f"X_test_{ngram}g.csv"), index=False)
    y_train.to_csv(os.path.join(EXPORT_DIR, f"y_train_{ngram}g.csv"), index=False, header=False)
    y_test.to_csv(os.path.join(EXPORT_DIR, f"y_test_{ngram}g.csv"), index=False, header=False)
    print(f"[SAVE] feature_map_{ngram}g.csv / X_*_{ngram}g.csv / y_*_{ngram}g.csv")

    return X_train_df, X_test_df, y_train, y_test

def main():
    # Ingest and analysis
    print(f"\n[INGEST] Scanning for .opcode files under: {DATA_DIR}")
    # records Filename, File size, APT label, File type,Opcode statistics, unique count etc
    # docs are concat opcode string for processing
    # paths is for filepath/filename related data to query
    records, docs, paths = [], [], []
    all_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.opcode"), recursive=True)
    print(f"[INGEST] Found {len(all_paths)} .opcode files.")

    for i, fp in enumerate(all_paths, 1):
        try:
            ops = read_opcodes(fp)
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")
            continue

        apt   = parse_meta(fp)
        ftype = get_filetype(fp)
        size_b = os.path.getsize(fp)

        rec = {
            "path": fp,
            "filename": os.path.basename(fp),
            "apt": apt,
            "filetype": ftype,
            "size_bytes": size_b,
            "opcode_count": len(ops),
            "unique_opcodes": len(set(ops)),
        }
        records.append(rec)
        docs.append(" ".join(ops))
        paths.append(fp)

        if i % 50 == 0:
            print(f"[INGEST] Processed {i}/{len(all_paths)} files...")

    df = pd.DataFrame(records)
    df["doc"] = docs

    compute_ngram_stats(docs, ngram_range=(1, 1), export_name="unigrams", export_dir=EXPORT_DIR)
    compute_ngram_stats(docs, ngram_range=(2, 2), export_name="bigrams", export_dir=EXPORT_DIR)

    print("\n[INGEST] Summary head (first 3 rows):")
    print(df.head(3)[["filename","apt","filetype","size_bytes","opcode_count","unique_opcodes"]])

    # Empty guard
    if df.empty:
        print("\n[ERROR] No files ingested. Check DATA_DIR path or file extensions.")
        raise SystemExit(2)

    # Save raw summary
    df.to_csv(os.path.join(EXPORT_DIR, "file_summary.csv"), index=False)
    print(f"[SAVE] file_summary.csv -> {EXPORT_DIR}")

    # Sanity check for unique opcode counts
    mismatches = []
    for i, row in df.iterrows():
        ops_direct = read_opcodes(row["path"])
        u_direct = len(set(ops_direct))
        if u_direct != row["unique_opcodes"]:
            mismatches.append((row["filename"], u_direct, row["unique_opcodes"]))
    print(f"[CHECK] Unique opcode mismatches (first 10): {mismatches[:10]}")

    # Counts
    by_apt = df["apt"].value_counts()
    by_apt.to_csv(os.path.join(EXPORT_DIR, "count_by_apt.csv"))
    print(f"[SAVE] count_by_apt.csv -> {EXPORT_DIR}")
    print(f"[INFO] Top APT counts:\n {by_apt.head(15).to_string()}")

    by_type = df["filetype"].value_counts()
    by_type.to_csv(os.path.join(EXPORT_DIR, "count_by_filetype.csv"))
    print(f"[SAVE] count_by_filetype.csv -> {EXPORT_DIR}")
    print(f"[INFO] Filetype distribution:\n {by_type.to_string()}")

    # Analysis plots
    try:
        plt.figure(figsize=(10,5))
        by_apt.head(20).plot(kind="bar")
        plt.title("Samples per APT (Top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(EXPORT_DIR, "plot_samples_per_apt.png"), dpi=150)
        plt.close()
        print(f"[SAVE] plot_samples_per_apt.png -> {EXPORT_DIR}")

        plt.figure(figsize=(8,4))
        df["opcode_count"].hist(bins=40)
        plt.title("Distribution of opcode counts per file")
        plt.xlabel("opcode_count")
        plt.tight_layout()
        plt.savefig(os.path.join(EXPORT_DIR, "plot_opcode_count_hist.png"), dpi=150)
        plt.close()
        print(f"[SAVE] plot_opcode_count_hist.png -> {EXPORT_DIR}")
    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")

    print(f"\n[INGEST] Done. CSVs saved under: {EXPORT_DIR}")

    # -----------------------------
    # A) RARE CLASS MERGE (≤5 -> OTHER)
    # -----------------------------
    MIN_SAMPLES = 5
    counts_raw = df["apt"].value_counts()
    rare_labels = counts_raw[counts_raw <= MIN_SAMPLES].index
    if len(rare_labels):
        df["apt"] = df["apt"].where(~df["apt"].isin(rare_labels), "OTHER")

    print("\n[CLASS BALANCE] After rare merge")
    print(df["apt"].value_counts().sort_values(ascending=False).to_string())

    # Build both datasets (1g & 2g)
    X_train_1g, X_test_1g, y_train_1g, y_test_1g = build_ml_dataframe(df, ngram=1)
    X_train_2g, X_test_2g, y_train_2g, y_test_2g = build_ml_dataframe(df, ngram=2)

    print("\n[BUILD] ML dataframes ready:")
    print(f"      1g: {X_train_1g.shape} {X_test_1g.shape}")
    print(f"      2g: {X_train_2g.shape} {X_test_2g.shape}")

if __name__ == "__main__":
    main()