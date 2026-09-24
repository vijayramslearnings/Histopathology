"""Patient-level split validation and generation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "ood_test": "ood_test",
}


def _normalize_split_names(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(_SPLIT_ALIASES))
    if unknown:
        raise ValueError(
            "Unsupported split name(s): "
            f"{unknown}. Use train, val/validation, or test."
        )
    return normalized.map(_SPLIT_ALIASES)


def validate_patient_disjointness(metadata: pd.DataFrame) -> None:
    if "split" not in metadata.columns:
        return
    grouped = metadata.dropna(subset=["split"]).groupby("split")["patient_id"].apply(set)
    names = list(grouped.index)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = grouped[first].intersection(grouped[second])
            if overlap:
                examples = ", ".join(sorted(str(value) for value in overlap)[:5])
                raise ValueError(
                    f"Patient leakage detected between '{first}' and '{second}': {examples}"
                )


def _group_split(
    metadata: pd.DataFrame,
    train_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_indices, other_indices = next(
        splitter.split(metadata, groups=metadata["patient_id"])
    )
    return metadata.iloc[train_indices].copy(), metadata.iloc[other_indices].copy()


def make_patient_splits(
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return train/val/test partitions with zero patient overlap."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")
    patient_ids = metadata["patient_id"].fillna("").astype(str).str.strip()
    if patient_ids.eq("").any() or patient_ids.str.lower().isin({"nan", "none"}).any():
        raise ValueError("Every metadata row must have a patient_id for safe splitting.")

    metadata = metadata.reset_index(drop=True)
    validate_patient_disjointness(metadata)
    provided = (
        "split" in metadata.columns
        and metadata["split"].notna().all()
        and metadata["split"].astype(str).str.strip().ne("").all()
    )
    if provided:
        metadata = metadata.copy()
        metadata["split"] = _normalize_split_names(metadata["split"])
        splits = {
            name: metadata[metadata["split"].astype(str).str.lower() == name].copy()
            for name in ("train", "val", "test", "ood_test")
        }
        if splits["train"].empty or splits["test"].empty:
            raise ValueError("A supplied split column must contain both train and test rows.")
        if splits["val"].empty:
            train, val = _group_split(
                splits["train"],
                train_size=1.0 - val_ratio / (train_ratio + val_ratio),
                seed=seed,
            )
            splits["train"], splits["val"] = train, val
        if sum(len(frame) for frame in splits.values()) != len(metadata):
            raise ValueError("Some metadata rows were not assigned to a recognized split.")
        validate_patient_disjointness(
            pd.concat(
                [
                    frame.assign(split=name)
                    for name, frame in splits.items()
                    if not frame.empty
                ],
                ignore_index=True,
            )
        )
        return splits

    train_plus_val, test = _group_split(metadata, train_ratio + val_ratio, seed)
    train, val = _group_split(
        train_plus_val,
        train_size=train_ratio / (train_ratio + val_ratio),
        seed=seed + 1,
    )
    return {"train": train, "val": val, "test": test}