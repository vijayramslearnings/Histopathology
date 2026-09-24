from __future__ import annotations

import pandas as pd
import pytest

from backend.dataset.dataset import build_label_maps


def _metadata(subtypes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": ["a.png"] * len(subtypes),
            "patient_id": [f"P{index}" for index in range(len(subtypes))],
            "slide_id": [f"S{index}" for index in range(len(subtypes))],
            "cancer_type": ["BRCA", "LUAD", "GBM"][: len(subtypes)],
            "subtype": subtypes,
        }
    )


def test_subtype_head_is_disabled_when_labels_are_unavailable() -> None:
    labels = build_label_maps(_metadata(["", "", ""]))
    assert labels.cancer_classes == ["BRCA", "GBM", "LUAD"]
    assert labels.subtype_classes == []
    assert labels.has_subtypes is False


def test_subtype_classes_are_read_from_metadata() -> None:
    labels = build_label_maps(_metadata(["basal", "", "proneural"]))
    assert labels.subtype_classes == ["basal", "proneural"]
    assert labels.has_subtypes is True


def test_configured_class_mapping_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_label_maps(
            _metadata(["", "", ""]),
            cancer_mapping={"BRCA": 0, "LUAD": 2, "GBM": 3},
        )