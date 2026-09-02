# scripts_prepare_data.py
"""
Create a small stratified sample and a compressed subset from a large CSV.
Writes:
  - data/sample_base.csv        (small stratified sample for repo)
  - data/base_subset.parquet   (larger compressed subset, optional to keep locally)
  - data/base_subset.csv.gz    (gzipped CSV subset, optional)
Adjust SAMPLE_ROWS and SUBSET_ROWS as needed.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# ---------- Configuration ----------
SRC = Path(r"C:\Users\KEKELI\OneDrive\Desktop\PythonProg\Base.csv")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_ROWS = 10000        # rows to keep in sample_base.csv
SUBSET_ROWS = 50000        # rows to write to compressed subset files
RANDOM_STATE = 42
CHUNK_SIZE = 200_000      # used only if fallback to chunked processing is needed

# ---------- Helpers ----------
def downcast_numeric(df_in):
    df = df_in.copy()
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        # skip columns with many NaNs that would break is_integer_dtype
        try:
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")
        except Exception:
            # if any issue, leave column as-is
            continue
    return df

def convert_categories(df_in, max_unique=100):
    df = df_in.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique(dropna=False) <= max_unique:
            df[col] = df[col].astype("category")
    return df

# ---------- Core logic ----------
def create_sample_and_subset(df):
    total_rows = len(df)
    target_rows = min(SAMPLE_ROWS, total_rows)

    if "fraud_bool" in df.columns:
        counts = df["fraud_bool"].value_counts().sort_index()
        alloc = (counts / counts.sum() * target_rows).astype(int)
        alloc = alloc.clip(lower=1)
        alloc = alloc.where(alloc <= counts, counts)

        deficit = target_rows - alloc.sum()
        if deficit > 0:
            spare = (counts - alloc)
            for g in spare[spare > 0].index:
                if deficit <= 0:
                    break
                add = min(spare[g], deficit)
                alloc[g] += add
                deficit -= add

        if alloc.sum() < target_rows:
            sample = df.sample(n=target_rows, random_state=RANDOM_STATE)
        else:
            parts = []
            for g, n in alloc.items():
                parts.append(df[df["fraud_bool"] == g].sample(n=n, random_state=RANDOM_STATE))
            sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        sample = df.sample(n=target_rows, random_state=RANDOM_STATE)

    # create a larger random subset for local use
    subset_rows = min(SUBSET_ROWS, total_rows)
    subset = df.sample(n=subset_rows, random_state=RANDOM_STATE)

    # downcast and convert categories
    sample = downcast_numeric(convert_categories(sample))
    subset = downcast_numeric(convert_categories(subset))

    return sample, subset

def main():
    if not SRC.exists():
        print(f"ERROR: Source file not found at: {SRC}", file=sys.stderr)
        sys.exit(2)

    # Try full read first; if memory error occurs, fallback to chunked approach
    try:
        print("Reading full CSV into memory...")
        df = pd.read_csv(SRC)
        sample, subset = create_sample_and_subset(df)
    except MemoryError:
        print("MemoryError reading full CSV. Falling back to chunked processing...")
        # Chunked approach: compute class counts, then sample per class across chunks
        # First pass: compute counts and collect indices for sampling
        counts = {}
        total_rows = 0
        for chunk in pd.read_csv(SRC, chunksize=CHUNK_SIZE):
            total_rows += len(chunk)
            if "fraud_bool" in chunk.columns:
                vc = chunk["fraud_bool"].value_counts().to_dict()
                for k, v in vc.items():
                    counts[k] = counts.get(k, 0) + v

        target_rows = min(SAMPLE_ROWS, total_rows)
        if "fraud_bool" in counts:
            # compute allocation safely
            counts_series = pd.Series(counts)
            alloc = (counts_series / counts_series.sum() * target_rows).astype(int).clip(lower=1)
            alloc = alloc.where(alloc <= counts_series, counts_series)
            deficit = target_rows - alloc.sum()
            if deficit > 0:
                spare = (counts_series - alloc)
                for g in spare[spare > 0].index:
                    if deficit <= 0:
                        break
                    add = min(spare[g], deficit)
                    alloc[g] += add
                    deficit -= add
        else:
            alloc = None

        # Second pass: sample from chunks according to allocation
        parts = {}
        subset_parts = []
        rng = np.random.RandomState(RANDOM_STATE)
        for chunk in pd.read_csv(SRC, chunksize=CHUNK_SIZE):
            # build subset parts
            subset_parts.append(chunk.sample(frac=min(1, SUBSET_ROWS / total_rows), random_state=rng))
            if alloc is None:
                # no stratify column, sample proportionally from chunk
                pass
            else:
                for g, n_needed in alloc.items():
                    if n_needed <= 0:
                        continue
                    rows_in_chunk = chunk[chunk["fraud_bool"] == g]
                    if len(rows_in_chunk) == 0:
                        continue
                    take = min(len(rows_in_chunk), n_needed)
                    sel = rows_in_chunk.sample(n=take, random_state=rng)
                    parts.setdefault(g, []).append(sel)
                    alloc[g] -= take

        # combine parts to form sample
        if alloc is None:
            # fallback: sample randomly from concatenated subset of chunks
            all_chunks = pd.concat(subset_parts)
            sample = all_chunks.sample(n=target_rows, random_state=RANDOM_STATE)
        else:
            collected = []
            for g, lst in parts.items():
                if lst:
                    collected.append(pd.concat(lst))
            if collected:
                sample = pd.concat(collected).sample(n=target_rows, random_state=RANDOM_STATE).reset_index(drop=True)
            else:
                # last resort
                all_chunks = pd.concat(subset_parts)
                sample = all_chunks.sample(n=target_rows, random_state=RANDOM_STATE)

        subset = pd.concat(subset_parts).sample(n=min(SUBSET_ROWS, total_rows), random_state=RANDOM_STATE)
        sample = downcast_numeric(convert_categories(sample))
        subset = downcast_numeric(convert_categories(subset))

    # Save outputs
    sample_csv = OUT_DIR / "sample_base.csv"
    subset_parquet = OUT_DIR / "base_subset.parquet"
    subset_csv_gz = OUT_DIR / "base_subset.csv.gz"

    print("Writing files...")
    sample.to_csv(sample_csv, index=False)
    try:
        subset.to_parquet(subset_parquet, index=False, compression="snappy")
    except Exception:
        # if pyarrow/fastparquet not available, skip parquet
        print("Warning: parquet write failed; skipping parquet output")
    subset.to_csv(subset_csv_gz, index=False, compression="gzip")

    print("Wrote:", sample_csv, subset_parquet, subset_csv_gz)

if __name__ == "__main__":
    main()
