"""Metadata-driven histopathology dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import BaselinePreprocessor, MultiScalePreprocessor

REQUIRED_COLUMNS = {"image_path", "patient_id", "slide_id", "cancer_type", "subtype"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    metadata = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    metadata = metadata.copy()
    metadata["image_path"] = metadata["image_path"].fillna("").astype(str).str.strip()
    metadata["patient_id"] = metadata["patient_id"].fillna("").astype(str).str.strip()
    metadata["slide_id"] = metadata["slide_id"].fillna("").astype(str).str.strip()
    metadata["cancer_type"] = metadata["cancer_type"].fillna("").astype(str).str.strip()
    metadata["subtype"] = metadata["subtype"].fillna("").astype(str).str.strip()
    for column in ("image_path", "patient_id", "slide_id", "cancer_type"):
        invalid = metadata[column].eq("") | metadata[column].str.lower().isin({"nan", "none"})
        if invalid.any():
            raise ValueError(f"Metadata contains an empty {column}.")
    metadata.attrs["metadata_dir"] = str(path.parent.resolve())
    return metadata


@dataclass(frozen=True)
class LabelMaps:
    cancer_to_index: dict[str, int]
    subtype_to_index: dict[str, int]
    subtype_specs: dict[str, dict[str, int]] = None

    @property
    def cancer_classes(self) -> list[str]:
        return [label for label, _ in sorted(self.cancer_to_index.items(), key=lambda item: item[1])]

    @property
    def subtype_classes(self) -> list[str]:
        return [label for label, _ in sorted(self.subtype_to_index.items(), key=lambda item: item[1])]

    @property
    def has_subtypes(self) -> bool:
        return bool(self.subtype_to_index)


def build_label_maps(
    metadata: pd.DataFrame,
    cancer_mapping: Optional[Mapping[str, int | str]] = None,
) -> LabelMaps:
    if cancer_mapping:
        ordered = sorted(cancer_mapping.items(), key=lambda item: int(item[1]))
        indices = [int(index) for _, index in ordered]
        if indices != list(range(len(indices))):
            raise ValueError("cancer_mapping indices must be contiguous and start at zero.")
        cancer_to_index = {str(label): int(index) for label, index in ordered}
        unknown = set(metadata[metadata["cancer_type"] != "OOD"]["cancer_type"]) - set(cancer_to_index)
        if unknown:
            raise ValueError(f"Metadata contains cancer classes absent from config: {sorted(unknown)}")
    else:
        in_dist = metadata[metadata["cancer_type"] != "OOD"]
        cancer_to_index = {
            label: index
            for index, label in enumerate(sorted(in_dist["cancer_type"].unique()))
        }

    subtype_values = sorted(
        value for value in metadata["subtype"].unique() if value and value.lower() not in {"nan", "none", "normal"}
    )
    subtype_to_index = {label: index for index, label in enumerate(subtype_values)}

    # Build cancer-specific subtype maps
    subtype_specs = {}
    for cancer in cancer_to_index:
        c_df = metadata[metadata["cancer_type"] == cancer]
        c_subtypes = sorted(v for v in c_df["subtype"].unique() if v and v.lower() not in {"nan", "none", "normal"})
        subtype_specs[cancer] = {slabel: sidx for sidx, slabel in enumerate(c_subtypes)}

    return LabelMaps(cancer_to_index, subtype_to_index, subtype_specs)


class HistopathologyDataset(Dataset[dict[str, Any]]):
    """Reads image paths and labels without ever mixing patients by itself."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        labels: LabelMaps,
        preprocessor: MultiScalePreprocessor | BaselinePreprocessor,
        model_name: str = "multiscale",
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.labels = labels
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.metadata_dir = Path(
            str(metadata.attrs.get("metadata_dir", Path.cwd()))
        ).resolve()

    def __len__(self) -> int:
        return len(self.metadata)

    def _image_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        if (Path.cwd() / path).exists():
            return Path.cwd() / path
        if (self.metadata_dir / path).exists():
            return self.metadata_dir / path
        if (self.metadata_dir.parent / path).exists():
            return self.metadata_dir.parent / path
        return path if path.is_absolute() else self.metadata_dir / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        image_path = self._image_path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for metadata row {index}: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.preprocessor(image)
        
        cancer_type = str(row["cancer_type"]).strip()
        subtype = str(row["subtype"]).strip()

        cancer_label = self.labels.cancer_to_index.get(cancer_type, -1)
        subtype_label = self.labels.subtype_to_index.get(subtype, -1)

        # Cancer-specific subtype labels
        subtype_targets = {}
        if self.labels.subtype_specs:
            for cname, smap in self.labels.subtype_specs.items():
                subtype_targets[cname] = smap.get(subtype, -1) if cancer_type == cname else -1

        return {
            "image": tensor,
            "cancer_label": cancer_label,
            "subtype_label": subtype_label,
            "subtype_targets": subtype_targets,
            "cancer_type": cancer_type,
            "patient_id": str(row["patient_id"]),
            "slide_id": str(row["slide_id"]),
            "image_path": str(image_path),
        }