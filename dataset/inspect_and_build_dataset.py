"""Dataset Inspection, Patient-Level Splitting & Report Generation Script.

Generates:
1. data/metadata.csv (Pathology dataset with patient_id, slide_id, cancer_type, subtype, split)
2. dataset_report.csv
3. dataset_report.json
4. label_distribution.png
5. config/labels.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt


def create_realistic_histopathology_patch(
    cancer_type: str,
    subtype: str,
    seed: int,
    size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """Renders synthetic tissue patch with H&E staining patterns tailored per cancer type."""
    rng = np.random.default_rng(seed)
    width, height = size

    # Base H&E Staining background (pinkish/purple matrix)
    if cancer_type == "BRCA":
        # Breast invasive ductal pattern (dense fibrous stroma + dark violet epithelial nests)
        bg_color = (235, 200, 220)
        stain_dark = (110, 40, 110)
        stain_light = (200, 150, 190)
    elif cancer_type == "LUAD":
        # Lung adenocarcinoma (glandular structures, alveolar spaces)
        bg_color = (240, 210, 225)
        stain_dark = (90, 30, 120)
        stain_light = (180, 130, 175)
    elif cancer_type == "GBM":
        # Glioblastoma (hypercellularity, pseudopalisading necrosis, vascular proliferation)
        bg_color = (220, 185, 210)
        stain_dark = (80, 20, 100)
        stain_light = (160, 100, 160)
    else: # OOD / Normal
        bg_color = (245, 235, 240)
        stain_dark = (160, 140, 170)
        stain_light = (220, 210, 220)

    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    # Draw cellular nucleus clusters
    num_nuclei = rng.integers(80, 200) if cancer_type != "OOD" else rng.integers(10, 30)
    for _ in range(num_nuclei):
        cx = rng.integers(10, width - 10)
        cy = rng.integers(10, height - 10)
        r = rng.integers(3, 8)
        color = stain_dark if rng.random() > 0.3 else stain_light
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Apply subtle tissue blur & texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def generate_and_verify_dataset(
    output_dir: str | Path = "data",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generates verification dataset, enforces patient-level split, saves metadata & reports."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path("configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Subtype definition per cancer type
    cancer_subtypes = {
        "BRCA": ["IDC_LuminalA", "IDC_LuminalB", "TNBC"],
        "LUAD": ["Acinar_Predominant", "Papillary_Predominant"],
        "GBM": ["Mesenchymal", "Proneural"],
    }

    records = []
    patient_counter = 1
    slide_counter = 1

    # Patient distribution (30 patients BRCA, 25 LUAD, 20 GBM, 5 OOD)
    cohort_specs = [
        ("BRCA", 30, [0.50, 0.30, 0.20]),
        ("LUAD", 25, [0.60, 0.40]),
        ("GBM", 20, [0.55, 0.45]),
    ]

    for cancer_type, num_patients, subtype_probs in cohort_specs:
        subtypes = cancer_subtypes[cancer_type]
        for p_idx in range(num_patients):
            patient_id = f"PATIENT_{cancer_type}_{patient_counter:03d}"
            patient_counter += 1
            selected_subtype = rng.choice(subtypes, p=subtype_probs)

            # Each patient has 1-3 slides, 2-4 patches per slide
            num_slides = rng.integers(1, 4)
            for s_idx in range(num_slides):
                slide_id = f"SLIDE_{cancer_type}_{slide_counter:04d}"
                slide_counter += 1
                num_patches = rng.integers(2, 5)

                for patch_idx in range(num_patches):
                    img_filename = f"{patient_id}_{slide_id}_p{patch_idx}.png"
                    img_path = images_dir / img_filename

                    # Generate patch
                    patch_seed = seed + len(records) * 17
                    img = create_realistic_histopathology_patch(
                        cancer_type, selected_subtype, patch_seed
                    )
                    img.save(img_path)

                    records.append({
                        "image_path": str(img_path.relative_to(Path.cwd() if img_path.is_relative_to(Path.cwd()) else output_dir.parent)),
                        "patient_id": patient_id,
                        "slide_id": slide_id,
                        "cancer_type": cancer_type,
                        "subtype": selected_subtype,
                        "magnification": "20x",
                    })

    # Add 10 Out-Of-Distribution samples (Normal Tissue)
    for ood_idx in range(10):
        patient_id = f"PATIENT_OOD_{patient_counter:03d}"
        patient_counter += 1
        slide_id = f"SLIDE_OOD_{slide_counter:04d}"
        slide_counter += 1
        img_filename = f"{patient_id}_{slide_id}_p0.png"
        img_path = images_dir / img_filename
        img = create_realistic_histopathology_patch("OOD", "Normal", seed + 999 + ood_idx)
        img.save(img_path)
        records.append({
            "image_path": str(img_path.relative_to(Path.cwd() if img_path.is_relative_to(Path.cwd()) else output_dir.parent)),
            "patient_id": patient_id,
            "slide_id": slide_id,
            "cancer_type": "OOD",
            "subtype": "Normal",
            "magnification": "20x",
        })

    df = pd.DataFrame(records)

    # Perform Patient-Level Split (70% train, 15% val, 15% test) for in-distribution patients
    in_dist_df = df[df["cancer_type"] != "OOD"].copy()
    ood_df = df[df["cancer_type"] == "OOD"].copy()

    unique_patients = in_dist_df["patient_id"].unique()
    rng.shuffle(unique_patients)

    n_total_patients = len(unique_patients)
    n_train = int(0.70 * n_total_patients)
    n_val = int(0.15 * n_total_patients)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])

    # Assert 0% patient leakage
    assert train_patients.isdisjoint(val_patients), "Patient leakage detected between train and val!"
    assert train_patients.isdisjoint(test_patients), "Patient leakage detected between train and test!"
    assert val_patients.isdisjoint(test_patients), "Patient leakage detected between val and test!"

    def assign_split(pid: str) -> str:
        if pid in train_patients:
            return "train"
        if pid in val_patients:
            return "val"
        if pid in test_patients:
            return "test"
        return "ood_test"

    df["split"] = df["patient_id"].apply(assign_split)

    # Save data/metadata.csv
    metadata_csv_path = output_dir / "metadata.csv"
    df.to_csv(metadata_csv_path, index=False)
    print(f"[INFO] Generated metadata CSV: {metadata_csv_path} ({len(df)} samples)")

    # Save dataset_report.csv
    report_csv_path = Path("dataset_report.csv")
    df[["image_path", "patient_id", "slide_id", "cancer_type", "subtype", "split"]].to_csv(
        report_csv_path, index=False
    )

    # Build dataset_report.json summary
    class_counts = df[df["split"] != "ood_test"]["cancer_type"].value_counts().to_dict()
    subtype_counts = (
        df[df["split"] != "ood_test"]
        .groupby(["cancer_type", "subtype"])
        .size()
        .unstack(fill_value=0)
        .to_dict()
    )

    patient_counts_per_split = {
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "ood_patients": len(ood_df["patient_id"].unique()),
    }

    report_json = {
        "total_samples": len(df),
        "in_distribution_samples": len(in_dist_df),
        "ood_samples": len(ood_df),
        "patient_level_splitting": {
            "disjoint_guarantee": True,
            "patient_counts": patient_counts_per_split,
        },
        "cancer_class_distribution": class_counts,
        "subtype_distribution": subtype_counts,
    }

    report_json_path = Path("dataset_report.json")
    report_json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    # Generate label_distribution.png figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cancer distribution plot
    counts = df["cancer_type"].value_counts()
    axes[0].bar(counts.index, counts.values, color=['#06b6d4', '#3b82f6', '#6366f1', '#a855f7'])
    axes[0].set_title("Cancer Type Sample Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Cancer Label")
    axes[0].set_ylabel("Number of Samples")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Subtype distribution per cancer
    sub_df = df[df["cancer_type"].isin(["BRCA", "LUAD", "GBM"])].groupby(["cancer_type", "subtype"]).size().reset_index(name="count")
    sub_labels = [f"{row['cancer_type']}\n{row['subtype']}" for _, row in sub_df.iterrows()]
    axes[1].barh(sub_labels, sub_df["count"], color='#0284c7')
    axes[1].set_title("Hierarchical Subtype Sample Counts", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sample Count")
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=300)
    plt.close()
    print("[INFO] Generated label distribution plot: label_distribution.png")

    # Generate config/labels.json with explicit cancer & cancer-specific subtype mappings
    labels_config = {
        "CANCER_CLASSES": {
            "BRCA": 0,
            "LUAD": 1,
            "GBM": 2
        },
        "SUBTYPE_CLASSES": {
            "BRCA": {
                "IDC_LuminalA": 0,
                "IDC_LuminalB": 1,
                "TNBC": 2
            },
            "LUAD": {
                "Acinar_Predominant": 0,
                "Papillary_Predominant": 1
            },
            "GBM": {
                "Mesenchymal": 0,
                "Proneural": 1
            }
        },
        "SUBTYPE_AVAILABLE": {
            "BRCA": True,
            "LUAD": True,
            "GBM": True
        }
    }

    labels_json_path = config_dir / "labels.json"
    labels_json_path.write_text(json.dumps(labels_config, indent=2), encoding="utf-8")
    print(f"[INFO] Generated label taxonomy config: {labels_json_path}")

    return df, report_json


if __name__ == "__main__":
    generate_and_verify_dataset()
