from __future__ import annotations

import pandas as pd
import pytest

from backend.dataset.splitting import (
    make_patient_splits,
    validate_patient_disjointness,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [f"images/{index}.png" for index in range(8)],
            "patient_id": [f"P{index}" for index in range(8)],
            "slide_id": [f"S{index}" for index in range(8)],
            "cancer_type": ["BRCA", "LUAD", "GBM", "BRCA"] * 2,
            "subtype": [""] * 8,
        }
    )


def test_generated_splits_are_patient_disjoint() -> None:
    splits = make_patient_splits(
        _metadata(), train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7
    )
    patient_sets = [set(frame["patient_id"]) for frame in splits.values()]
    assert all(left.isdisjoint(right) for index, left in enumerate(patient_sets) for right in patient_sets[index + 1 :])
    assert sum(len(frame) for frame in splits.values()) == 8


def test_supplied_split_with_shared_patient_is_rejected() -> None:
    metadata = _metadata()
    metadata["split"] = ["train", "train", "val", "test", "test", "train", "val", "test"]
    metadata.loc[metadata["patient_id"] == "P0", "patient_id"] = "SHARED"
    metadata.loc[metadata["patient_id"] == "P3", "patient_id"] = "SHARED"
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_disjointness(metadata)


def test_missing_patient_id_is_rejected() -> None:
    metadata = _metadata()
    metadata.loc[0, "patient_id"] = ""
    with pytest.raises(ValueError, match="patient_id"):
        make_patient_splits(metadata)


def test_split_aliases_are_normalized_without_dropping_rows() -> None:
    metadata = _metadata()
    metadata["split"] = [
        "training",
        "train",
        "validation",
        "valid",
        "testing",
        "test",
        "train",
        "test",
    ]
    splits = make_patient_splits(metadata, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert sum(len(frame) for frame in splits.values()) == len(metadata)


def test_unknown_split_name_is_rejected() -> None:
    metadata = _metadata()
    metadata["split"] = ["train"] * 7 + ["holdout"]
    with pytest.raises(ValueError, match="Unsupported split"):
        make_patient_splits(metadata)