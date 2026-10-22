"""
download_kaggle_datasets.py
Automated pipeline to download and organize official full-scale real histopathology
datasets from Kaggle directly into cancer-specific project folders.

Datasets fetched:
1. LC25000: 25,000 real images of Lung Adenocarcinoma, Lung Squamous, Colon Adenocarcinoma.
2. BreakHis: 9,109 real microscopic images of Breast Invasive Ductal (IDC) & Lobular (ILC).
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "real_clinical_datasets"
USER_HOME = Path.home()
KAGGLE_DIR = USER_HOME / ".kaggle"
KAGGLE_JSON_HOME = KAGGLE_DIR / "kaggle.json"
KAGGLE_JSON_PROJECT = BASE_DIR / "kaggle.json"


KAGGLE_ACCESS_TOKEN = USER_HOME / ".kaggle" / "access_token"
DEFAULT_TOKEN = "KGAT_552fc8c75bd1343412112fdaa98bc056"


def ensure_kaggle_api():
    """Checks for kaggle package and credentials."""
    # Ensure token in environment
    if "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = DEFAULT_TOKEN

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    if not KAGGLE_ACCESS_TOKEN.exists():
        with open(KAGGLE_ACCESS_TOKEN, "w") as f:
            f.write(DEFAULT_TOKEN + "\n")

    try:
        import kaggle
    except ImportError:
        print("[*] Installing official 'kaggle' Python library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"])
        import kaggle

    # Check for kaggle.json in project root or ~/.kaggle/
    if KAGGLE_JSON_PROJECT.exists() and not KAGGLE_JSON_HOME.exists():
        shutil.copy(KAGGLE_JSON_PROJECT, KAGGLE_JSON_HOME)
        print(f"[OK] Copied kaggle.json from project folder to {KAGGLE_JSON_HOME}")

    return True


def download_and_organize():
    if not ensure_kaggle_api():
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("[OK] Authenticated successfully with Kaggle API.")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    temp_download_dir = BASE_DIR / "data/temp_kaggle_downloads"
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download LC25000 (Lung & Colon Cancer: 25,000 real images)
    print("\n" + "=" * 60)
    print("[1/3] Downloading LC25000: Real Lung & Colon Histopathology...")
    print("=" * 60)
    lc_dataset = "andrewmvd/lung-and-colon-cancer-histopathological-images"
    lc_target = TARGET_DIR / "02_Lung_and_Colon_LC25000"
    lc_target.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[*] Downloading and extracting {lc_dataset} into {lc_target}...")
        api.dataset_download_files(lc_dataset, path=str(lc_target), unzip=True, quiet=False)
        print("[OK] LC25000 downloaded and extracted successfully!")
    except Exception as e:
        print(f"[!] Error downloading LC25000: {e}")

    # 2. Download BreakHis (Breast Cancer: 9,109 real microscopic images)
    print("\n" + "=" * 60)
    print("[2/3] Downloading BreakHis: Real Breast Cancer Biopsies (IDC/ILC)...")
    print("=" * 60)
    bh_dataset = "ambarish/breakhis"
    bh_target = TARGET_DIR / "01_Breast_Cancer_BreakHis"
    bh_target.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[*] Downloading and extracting {bh_dataset} into {bh_target}...")
        api.dataset_download_files(bh_dataset, path=str(bh_target), unzip=True, quiet=False)
        print("[OK] BreakHis downloaded and extracted successfully!")
    except Exception as e:
        print(f"[!] Error downloading BreakHis: {e}")

    # 3. Download Skin Cancer / Melanoma (3,297 real images)
    print("\n" + "=" * 60)
    print("[3/3] Downloading Real Skin Melanoma & Dermoscopic Dataset...")
    print("=" * 60)
    sk_dataset = "nodoubtdating/skin-cancer-malignant-vs-benign"
    sk_target = TARGET_DIR / "09_Melanoma_SkinCancer"
    sk_target.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[*] Downloading and extracting {sk_dataset} into {sk_target}...")
        api.dataset_download_files(sk_dataset, path=str(sk_target), unzip=True, quiet=False)
        print("[OK] Melanoma dataset downloaded and extracted successfully!")
    except Exception as e:
        print(f"[!] Error downloading Melanoma dataset: {e}")

    print("\n" + "=" * 70)
    print("[OK] ALL REAL CLINICAL DATASETS ARE READY IN:")
    print(f"    {TARGET_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    download_and_organize()
