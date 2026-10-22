"""
select_pilot.py
Selects a balanced pilot cohort of ~20 patients per cancer type (~100 slides total)
balanced across subtypes and generates 5-fold cross-validation patient-level splits
stratified by cancer type and subtypes.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42
TARGET_PILOT_PER_CANCER = 20

def create_strat_label(row):
    """Create a composite stratification label combining cancer_type and available subtypes."""
    ctype = str(row["cancer_type"])
    if ctype == "BRCA":
        subtype = str(row["brca_pam50"]) if pd.notna(row["brca_pam50"]) and row["brca_pam50"] != "Normal-like" else str(row["brca_histology"])
    elif ctype == "LUNG":
        subtype = str(row["lung_histology"])
    elif ctype == "KIDNEY":
        subtype = str(row["kidney_histology"])
    elif ctype == "COLON":
        subtype = str(row["colon_msi"])
    elif ctype == "GLIOMA":
        subtype = f"{row['glioma_grade']}_{row['glioma_idh']}"
    else:
        subtype = "unknown"
    return f"{ctype}_{subtype}"

def select_pilot():
    np.random.seed(RANDOM_SEED)
    master_path = Path("data/labels/master_labels.csv")
    if not master_path.exists():
        raise FileNotFoundError("data/labels/master_labels.csv not found.")

    df = pd.read_csv(master_path)
    
    # Exclude Normal-like brca_pam50 per counting rule
    df = df[df["brca_pam50"] != "Normal-like"].copy()

    df["strat_label"] = df.apply(create_strat_label, axis=1)

    pilot_dfs = []
    cancer_types = ["BRCA", "LUNG", "KIDNEY", "COLON", "GLIOMA"]

    for ctype in cancer_types:
        subset = df[df["cancer_type"] == ctype].copy()
        
        # Balance sample across strata within this cancer type
        strata = subset["strat_label"].value_counts()
        n_strata = len(strata)
        base_per_stratum = max(1, TARGET_PILOT_PER_CANCER // n_strata)

        selected_pids = []
        # Sample proportionally or evenly across strata
        for strat_val in strata.index:
            stratum_pool = subset[subset["strat_label"] == strat_val]
            n_take = min(len(stratum_pool), base_per_stratum)
            chosen = stratum_pool.sample(n=n_take, random_state=RANDOM_SEED)
            selected_pids.extend(chosen["patient_id"].tolist())

        # If we need more to reach 20, sample from remaining
        remaining = subset[~subset["patient_id"].isin(selected_pids)]
        needed = TARGET_PILOT_PER_CANCER - len(selected_pids)
        if needed > 0 and len(remaining) > 0:
            extra = remaining.sample(n=min(needed, len(remaining)), random_state=RANDOM_SEED)
            selected_pids.extend(extra["patient_id"].tolist())

        # If slightly over, cap at 20
        selected_pids = selected_pids[:TARGET_PILOT_PER_CANCER]
        pilot_dfs.append(subset[subset["patient_id"].isin(selected_pids)])

    pilot_df = pd.concat(pilot_dfs, ignore_index=True)
    
    # Generate 5-fold stratified cross validation splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pilot_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(pilot_df, pilot_df["cancer_type"])):
        pilot_df.iloc[val_idx, pilot_df.columns.get_loc("fold")] = fold

    # Save outputs
    output_pilot_path = Path("data/pilot_cohort.csv")
    pilot_df.to_csv(output_pilot_path, index=False)
    print(f"Saved pilot cohort ({len(pilot_df)} patients) to {output_pilot_path}")

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_csv = splits_dir / "pilot_5fold_splits.csv"
    pilot_df[["patient_id", "project", "cancer_type", "tss", "fold"]].to_csv(splits_csv, index=False)
    print(f"Saved 5-fold cross-validation splits to {splits_csv}")

    # Breakdown report
    print("\nPilot Cohort Summary:")
    print(pilot_df.groupby(["cancer_type", "fold"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    select_pilot()
