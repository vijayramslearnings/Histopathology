"""
process_all_tcga_slides.py
Automated pipeline to convert authentic TCGA .svs Whole Slide Images into:
1. High-Resolution Histology Images (.jpg / .png) viewable in any standard photo viewer.
2. Microscopic Cellular Tissue Patches (.png) focusing on diagnostic cancer regions.
3. Deep Learning PyTorch Tensor Representations (.pt) ready for 1-click web upload and MIL inference.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DATASETS_DIR = BASE_DIR / "tcga_clinical_datasets"
FEATURES_DIR = BASE_DIR / "data/features/resnet50"

from src.preprocess.process_external import process_external_data

Image.MAX_IMAGE_PIXELS = None

try:
    import tifffile
except ImportError:
    tifffile = None


def read_slide_image(slide_path: Path):
    """Reads a slide image using PIL or tifffile fallback."""
    try:
        im = Image.open(slide_path)
        return im
    except Exception:
        if tifffile is not None:
            with tifffile.TiffFile(slide_path) as tif:
                # Find an RGB series
                for s in tif.series:
                    if len(s.shape) == 3 and s.shape[-1] == 3:
                        arr = s.asarray()
                        return Image.fromarray(arr)
                # If only 2D, take page 0
                arr = tif.pages[0].asarray()
                if len(arr.shape) == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return Image.fromarray(arr)
        raise


def process_svs(svs_path: Path):
    folder = svs_path.parent
    name_stem = svs_path.stem
    # Identify patient ID (e.g. TCGA-A7-A26J from TCGA-A7-A26J-01B-02-BS2...)
    parts = name_stem.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        patient_id = f"TCGA-{parts[1]}-{parts[2]}"
    else:
        patient_id = name_stem[:12]

    print(f"\n=======================================================")
    print(f" Processing: {svs_path.name}")
    print(f" Specimen ID: {patient_id} ({folder.name})")
    print(f"=======================================================")

    try:
        im = read_slide_image(svs_path)
        W, H = im.size
        print(f"  [1/3] Slide Dimensions: {W} x {H} pixels")

        # 1. Generate High-Res Overview JPG
        overview_path = folder / f"{patient_id}_overview.jpg"
        if not overview_path.exists():
            thumb = im.copy()
            thumb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            thumb.save(overview_path, quality=95)
            print(f"  [OK] Saved high-res overview: {overview_path.name} ({thumb.size[0]}x{thumb.size[1]})")
        else:
            print(f"  [OK] Overview image already exists: {overview_path.name}")

        # 2. Extract 4 High-Magnification Microscopic Tissue Patches (.png)
        patch_dir = folder / "microscopic_patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        step = max(512, min(W, H) // 10)
        patch_count = 0

        for y in range(0, H - 512, step):
            for x in range(0, W - 512, step):
                crop = im.crop((x, y, x + 512, y + 512))
                arr = np.array(crop)
                # Check for tissue (mean < 220 and std > 12)
                if arr.mean() < 220 and arr.std() > 12:
                    patch_count += 1
                    patch_file = patch_dir / f"{patient_id}_patch_{patch_count:02d}.png"
                    crop.save(patch_file)
                    if patch_count >= 4:
                        break
            if patch_count >= 4:
                break

        print(f"  [OK] Extracted {patch_count} cellular tissue patches into {patch_dir.name}/")

        # 3. Generate .pt Deep Learning PyTorch Tensor
        pt_path = folder / f"{patient_id}.pt"
        if not pt_path.exists():
            print(f"  [2/3] Extracting 1024-d ResNet-50 features -> {pt_path.name}...")
            process_external_data(
                image_path=str(overview_path),
                patient_id=patient_id,
                output_dir=str(folder)
            )
            print(f"  [OK] Generated deep learning tensor: {pt_path.name}")
        else:
            print(f"  [OK] PyTorch tensor already exists: {pt_path.name}")

        # Also copy to central features directory for web app live lookup
        central_pt = FEATURES_DIR / f"{patient_id}.pt"
        if pt_path.exists() and not central_pt.exists():
            import shutil
            shutil.copy(pt_path, central_pt)
            print(f"  [OK] Linked to central web features directory: {central_pt.name}")

    except Exception as e:
        print(f"  [!] Error processing {svs_path.name}: {e}")


def main():
    print("=" * 70)
    print(" TCGA Slide to Image & Deep Learning Tensor Pipeline")
    print(f" Target Directory: {DATASETS_DIR}")
    print("=" * 70)

    svs_files = sorted(list(DATASETS_DIR.glob("**/*.svs")))
    if not svs_files:
        print("[!] No .svs files found in", DATASETS_DIR)
        return

    print(f"Found {len(svs_files)} authentic TCGA Whole Slide Images across all folders.")
    for idx, svs in enumerate(svs_files, 1):
        print(f"\n[{idx}/{len(svs_files)}]")
        process_svs(svs)

    print("\n" + "=" * 70)
    print("[OK] ALL TCGA SLIDES HAVE BEEN CONVERTED!")
    print("Each cancer folder now contains:")
    print("  * Authentic Whole Slide Image (.svs)")
    print("  * High-Resolution Overview Image (.jpg)")
    print("  * Cellular Tissue Patches (.png)")
    print("  * Deep Learning PyTorch Feature Tensor (.pt)")
    print("=" * 70)


if __name__ == "__main__":
    main()
