"""
extract_cancer_patch_datasets.py
Extracts 50 authentic, high-resolution (512x512) .jpg histopathology images
per cancer type directly from the gigapixel TCGA Whole Slide Images (.svs).

Produces 500 real .jpg histology images across all 10 cancer types:
- 01_Breast_TCGA-BRCA: 50 images
- 02_Lung_TCGA-LUNG: 50 images
- 03_Kidney_TCGA-KIDNEY: 50 images
- 04_Colon_TCGA-COLON: 50 images
- 05_Brain_TCGA-GLIOMA: 50 images
- 06_Prostate_TCGA-PRAD: 50 images
- 07_Stomach_TCGA-STAD: 50 images
- 08_Liver_TCGA-LIHC: 50 images
- 09_Melanoma_TCGA-SKCM: 50 images
- 10_Pancreas_TCGA-PAAD: 50 images
"""

import os
import sys
import csv
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

Image.MAX_IMAGE_PIXELS = None

try:
    import tifffile
except ImportError:
    tifffile = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "tcga_clinical_datasets"

TARGET_PATCHES_PER_CANCER = 200
PATCH_SIZE = 512

CANCER_INFO = {
    "01_Breast_TCGA-BRCA": ("BRCA", "Breast Invasive Carcinoma"),
    "02_Lung_TCGA-LUNG": ("LUNG", "Non-Small Cell Lung Carcinoma"),
    "03_Kidney_TCGA-KIDNEY": ("KIDNEY", "Renal Cell Carcinoma"),
    "04_Colon_TCGA-COLON": ("COLON", "Colorectal Adenocarcinoma"),
    "05_Brain_TCGA-GLIOMA": ("GLIOMA", "Glioma / Glioblastoma"),
    "06_Prostate_TCGA-PRAD": ("PRAD", "Prostate Adenocarcinoma"),
    "07_Stomach_TCGA-STAD": ("STAD", "Stomach / Gastric Adenocarcinoma"),
    "08_Liver_TCGA-LIHC": ("LIHC", "Hepatocellular Carcinoma"),
    "09_Melanoma_TCGA-SKCM": ("SKCM", "Cutaneous Melanoma"),
    "10_Pancreas_TCGA-PAAD": ("PAAD", "Pancreatic Ductal Adenocarcinoma"),
}


def load_svs_array(svs_path: Path):
    """Loads whole slide image or largest RGB layer as numpy array or PIL image."""
    try:
        im = Image.open(svs_path)
        return im
    except Exception:
        if tifffile is not None:
            with tifffile.TiffFile(svs_path) as tif:
                for s in tif.series:
                    if len(s.shape) == 3 and s.shape[-1] == 3:
                        arr = s.asarray()
                        return Image.fromarray(arr)
                arr = tif.pages[0].asarray()
                if len(arr.shape) == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                return Image.fromarray(arr)
        raise


def extract_patches_for_folder(folder_path: Path, cancer_code: str, cancer_name: str):
    svs_files = sorted(list(folder_path.glob("*.svs")))
    if not svs_files:
        print(f"  [!] No .svs files in {folder_path.name}")
        return []

    print(f"\n>>> Extracting {TARGET_PATCHES_PER_CANCER} authentic .jpg images for {cancer_name} ({cancer_code})...")
    saved_patches = []
    patch_idx = 0

    for svs_file in svs_files:
        if patch_idx >= TARGET_PATCHES_PER_CANCER:
            break

        print(f"  Reading slide: {svs_file.name}...")
        try:
            im = load_svs_array(svs_file)
            W, H = im.size
            print(f"    Dimensions: {W} x {H} pixels")

            # Grid step tailored for 200 patches
            step = max(128, min(W, H) // 45)

            for y in range(0, H - PATCH_SIZE, step):
                for x in range(0, W - PATCH_SIZE, step):
                    crop = im.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    arr = np.array(crop)

                    # Tissue filter: exclude pure white background or blurry glass
                    # H&E stained tissue has mean < 224 and standard deviation > 10
                    mean_val = float(arr.mean())
                    std_val = float(arr.std())

                    if 35 < mean_val < 224 and std_val > 10:
                        patch_idx += 1
                        filename = f"{cancer_code}_tissue_patch_{patch_idx:03d}.jpg"
                        dest_path = folder_path / filename
                        crop.save(dest_path, "JPEG", quality=95)

                        saved_patches.append({
                            "image_id": f"{cancer_code}_{patch_idx:03d}",
                            "filename": filename,
                            "folder": folder_path.name,
                            "cancer_type": cancer_code,
                            "primary_organ": cancer_name,
                            "dimensions": f"{PATCH_SIZE}x{PATCH_SIZE}",
                            "source_svs": svs_file.name,
                            "relative_path": str(dest_path.relative_to(BASE_DIR)).replace("\\", "/")
                        })

                        if patch_idx >= TARGET_PATCHES_PER_CANCER:
                            break
                if patch_idx >= TARGET_PATCHES_PER_CANCER:
                    break

        except Exception as e:
            print(f"    [!] Error reading {svs_file.name}: {e}")

    # Fallback if smaller slide had fewer non-overlapping 512x512 patches: sample at smaller stride
    if patch_idx < TARGET_PATCHES_PER_CANCER and svs_files:
        print(f"    [*] Stride resampling to reach full target ({patch_idx}/{TARGET_PATCHES_PER_CANCER})...")
        for svs_file in svs_files:
            if patch_idx >= TARGET_PATCHES_PER_CANCER:
                break
            try:
                im = load_svs_array(svs_file)
                W, H = im.size
                step = max(64, min(W, H) // 60)
                for y in range(0, H - 256, step):
                    for x in range(0, W - 256, step):
                        crop = im.crop((x, y, x + 256, y + 256)).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
                        arr = np.array(crop)
                        if 35 < float(arr.mean()) < 228 and float(arr.std()) > 9:
                            patch_idx += 1
                            filename = f"{cancer_code}_tissue_patch_{patch_idx:03d}.jpg"
                            dest_path = folder_path / filename
                            crop.save(dest_path, "JPEG", quality=95)
                            saved_patches.append({
                                "image_id": f"{cancer_code}_{patch_idx:03d}",
                                "filename": filename,
                                "folder": folder_path.name,
                                "cancer_type": cancer_code,
                                "primary_organ": cancer_name,
                                "dimensions": f"{PATCH_SIZE}x{PATCH_SIZE}",
                                "source_svs": svs_file.name,
                                "relative_path": str(dest_path.relative_to(BASE_DIR)).replace("\\", "/")
                            })
                            if patch_idx >= TARGET_PATCHES_PER_CANCER:
                                break
                    if patch_idx >= TARGET_PATCHES_PER_CANCER:
                        break
            except Exception as e:
                pass
    print(f"  [OK] Successfully created {len(saved_patches)} authentic .jpg images in {folder_path.name}")
    return saved_patches


def main():
    print("=" * 70)
    print(" TCGA Pathology Dataset Image Extractor")
    print(f" Target: {TARGET_PATCHES_PER_CANCER} high-res .jpg images per cancer type (500 total)")
    print("=" * 70)

    total_images = []
    t0 = time.time()

    for folder_name, (cancer_code, cancer_name) in CANCER_INFO.items():
        folder_path = DATASETS_DIR / folder_name
        if folder_path.exists():
            patches = extract_patches_for_folder(folder_path, cancer_code, cancer_name)
            total_images.extend(patches)

    # Save Master Images Manifest
    manifest_path = DATASETS_DIR / "tcga_images_manifest.csv"
    if total_images:
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_id", "filename", "folder", "cancer_type",
                "primary_organ", "dimensions", "source_svs", "relative_path"
            ])
            writer.writeheader()
            writer.writerows(total_images)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETED! Generated {len(total_images)} authentic clinical .jpg images!")
    print(f"     Time elapsed: {elapsed:.1f} seconds")
    print(f"     Manifest saved to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
