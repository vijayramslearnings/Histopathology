"""Dataset validation script for histopathology cancer datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from PIL import Image


SUPPORTED_CLASSES: Set[str] = {"BRCA", "LUAD", "GBM"}


def compute_file_hash(path: str | Path) -> str:
    """Compute MD5 hash of an image file to detect duplicate binaries."""
    hasher = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset_csv(
    csv_path: str | Path,
    base_dir: Optional[str | Path] = None,
    supported_classes: Set[str] = SUPPORTED_CLASSES,
) -> Dict[str, Any]:
    """Perform comprehensive validation of dataset CSV file and images."""
    csv_path = Path(csv_path)
    base_dir = Path(base_dir) if base_dir else csv_path.parent

    report: Dict[str, Any] = {
        "csv_path": str(csv_path),
        "total_rows": 0,
        "valid_samples": 0,
        "issues": [],
        "class_counts": {},
        "patient_split_counts": {},
        "missing_files": [],
        "corrupted_images": [],
        "invalid_labels": [],
        "duplicate_hashes": [],
        "duplicate_slide_ids": [],
        "leakage_warnings": [],
        "is_valid": True,
    }

    if not csv_path.exists():
        report["is_valid"] = False
        report["issues"].append(f"CSV file not found: {csv_path}")
        return report

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        report["is_valid"] = False
        report["issues"].append(f"Failed to parse CSV: {str(e)}")
        return report

    report["total_rows"] = len(df)
    required_cols = {"image_path", "patient_id", "slide_id", "cancer_type"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        report["is_valid"] = False
        report["issues"].append(f"Missing required columns in CSV: {missing_cols}")
        return report

    seen_hashes: Dict[str, str] = {}
    seen_slides: Set[str] = set()

    for idx, row in df.iterrows():
        rel_img_path = str(row["image_path"]).strip()
        patient_id = str(row["patient_id"]).strip()
        slide_id = str(row["slide_id"]).strip()
        cancer_type = str(row["cancer_type"]).strip().upper()

        # 1. Label check
        if cancer_type not in supported_classes:
            report["invalid_labels"].append({
                "row": idx,
                "label": cancer_type,
                "expected": list(supported_classes),
            })
        else:
            report["class_counts"][cancer_type] = report["class_counts"].get(cancer_type, 0) + 1

        # 2. Slide ID duplicate check
        if slide_id in seen_slides:
            report["duplicate_slide_ids"].append({"row": idx, "slide_id": slide_id})
        seen_slides.add(slide_id)

        # 3. File existence & image integrity check
        img_full_path = Path(rel_img_path)
        if not img_full_path.is_absolute():
            img_full_path = base_dir / rel_img_path

        if not img_full_path.exists():
            report["missing_files"].append({"row": idx, "path": str(img_full_path)})
            continue

        try:
            with Image.open(img_full_path) as img:
                img.verify()
            with Image.open(img_full_path) as img:
                w, h = img.size
                mode = img.mode
                if mode not in ("RGB", "RGBA"):
                    report["issues"].append(f"Row {idx}: Image mode '{mode}' is non-RGB.")
                if w < 100 or h < 100:
                    report["issues"].append(f"Row {idx}: Image resolution {w}x{h} is below minimum 100x100.")
        except Exception as err:
            report["corrupted_images"].append({"row": idx, "path": str(img_full_path), "error": str(err)})
            continue

        # 4. Duplicate content check via hash
        try:
            hval = compute_file_hash(img_full_path)
            if hval in seen_hashes:
                report["duplicate_hashes"].append({
                    "row": idx,
                    "path": str(img_full_path),
                    "duplicate_of": seen_hashes[hval],
                })
            else:
                seen_hashes[hval] = str(img_full_path)
        except Exception:
            pass

    # 5. Patient-level split leakage check
    if "split" in df.columns:
        patient_splits: Dict[str, Set[str]] = {}
        for _, row in df.iterrows():
            pid = str(row["patient_id"]).strip()
            sp = str(row["split"]).strip().lower()
            if pid and sp:
                if pid not in patient_splits:
                    patient_splits[pid] = set()
                patient_splits[pid].add(sp)

        for pid, splits in patient_splits.items():
            if len(splits) > 1:
                report["leakage_warnings"].append({
                    "patient_id": pid,
                    "overlapping_splits": list(splits),
                })

    # Summary issues count
    error_count = (
        len(report["missing_files"])
        + len(report["corrupted_images"])
        + len(report["invalid_labels"])
        + len(report["leakage_warnings"])
    )
    report["is_valid"] = error_count == 0
    report["valid_samples"] = report["total_rows"] - error_count

    return report


def save_validation_reports(
    report: Dict[str, Any],
    json_path: str | Path = "dataset_validation_report.json",
    txt_path: str | Path = "dataset_validation_report.txt",
) -> None:
    """Save validation report as structured JSON and human-readable TXT."""
    json_path = Path(json_path)
    txt_path = Path(txt_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "==================================================",
        "HISTOPATHOLOGY DATASET VALIDATION REPORT",
        "==================================================",
        f"CSV File: {report.get('csv_path')}",
        f"Overall Valid: {report.get('is_valid')}",
        f"Total Rows: {report.get('total_rows')}",
        f"Valid Samples: {report.get('valid_samples')}",
        "",
        "--- CLASS DISTRIBUTION ---",
    ]
    for cname, count in report.get("class_counts", {}).items():
        lines.append(f"  {cname}: {count}")

    lines.extend([
        "",
        "--- ISSUES & WARNINGS SUMMARY ---",
        f"  Missing Files: {len(report.get('missing_files', []))}",
        f"  Corrupted Images: {len(report.get('corrupted_images', []))}",
        f"  Invalid Labels: {len(report.get('invalid_labels', []))}",
        f"  Duplicate Content Hashes: {len(report.get('duplicate_hashes', []))}",
        f"  Patient Leakage Warnings: {len(report.get('leakage_warnings', []))}",
        "",
    ])

    if report.get("leakage_warnings"):
        lines.append("CRITICAL: Patient leakage detected across splits:")
        for lw in report["leakage_warnings"]:
            lines.append(f"  Patient {lw['patient_id']} present in: {lw['overlapping_splits']}")
        lines.append("")

    if report.get("duplicate_hashes"):
        lines.append("SUSPICIOUS: Duplicate image contents detected:")
        for dh in report["duplicate_hashes"][:5]:
            lines.append(f"  {dh['path']} duplicates {dh['duplicate_of']}")
        lines.append("")

    lines.append("Notice: Do NOT automatically change labels. Manually review flagged samples.")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
