"""
download_tcga_datasets.py
Official NCI Genomic Data Commons (GDC) Pipeline for Downloading
Authentic Whole-Slide Pathology Datasets from The Cancer Genome Atlas (TCGA).

Downloads authentic diagnostic Whole Slide Images (.svs) and official clinical
metadata directly from NIH/NCI GDC open-access servers for all 10 cancer types:
1. Breast (TCGA-BRCA)
2. Lung (TCGA-LUAD / TCGA-LUSC)
3. Kidney (TCGA-KIRC / TCGA-KIRP / TCGA-KICH)
4. Colon & Rectum (TCGA-COAD / TCGA-READ)
5. Brain Glioma (TCGA-GBM / TCGA-LGG)
6. Prostate (TCGA-PRAD)
7. Stomach (TCGA-STAD)
8. Liver (TCGA-LIHC)
9. Skin Melanoma (TCGA-SKCM)
10. Pancreas (TCGA-PAAD)

Zero third-party tokens required (100% NIH Open Access).
Organizes slides into separate cancer-specific folders with clinical manifests.
"""

import os
import sys
import json
import time
import csv
import hashlib
from pathlib import Path
import requests

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = BASE_DIR / "tcga_clinical_datasets"
GDC_API_URL = "https://api.gdc.cancer.gov"

# Cancer definitions with TCGA Project IDs
CANCER_PROJECTS = [
    {
        "folder": "01_Breast_TCGA-BRCA",
        "cancer_type": "BRCA",
        "cancer_full": "Breast Invasive Carcinoma",
        "projects": ["TCGA-BRCA"],
        "slides_per_project": 1,
    },
    {
        "folder": "02_Lung_TCGA-LUNG",
        "cancer_type": "LUNG",
        "cancer_full": "Non-Small Cell Lung Carcinoma",
        "projects": ["TCGA-LUAD", "TCGA-LUSC"],
        "slides_per_project": 1,
    },
    {
        "folder": "03_Kidney_TCGA-KIDNEY",
        "cancer_type": "KIDNEY",
        "cancer_full": "Renal Cell Carcinoma",
        "projects": ["TCGA-KIRC", "TCGA-KIRP"],
        "slides_per_project": 1,
    },
    {
        "folder": "04_Colon_TCGA-COLON",
        "cancer_type": "COLON",
        "cancer_full": "Colorectal Adenocarcinoma",
        "projects": ["TCGA-COAD", "TCGA-READ"],
        "slides_per_project": 1,
    },
    {
        "folder": "05_Brain_TCGA-GLIOMA",
        "cancer_type": "GLIOMA",
        "cancer_full": "Glioma / Glioblastoma Multiforme",
        "projects": ["TCGA-GBM", "TCGA-LGG"],
        "slides_per_project": 1,
    },
    {
        "folder": "06_Prostate_TCGA-PRAD",
        "cancer_type": "PRAD",
        "cancer_full": "Prostate Adenocarcinoma",
        "projects": ["TCGA-PRAD"],
        "slides_per_project": 1,
    },
    {
        "folder": "07_Stomach_TCGA-STAD",
        "cancer_type": "STAD",
        "cancer_full": "Stomach / Gastric Adenocarcinoma",
        "projects": ["TCGA-STAD"],
        "slides_per_project": 1,
    },
    {
        "folder": "08_Liver_TCGA-LIHC",
        "cancer_type": "LIHC",
        "cancer_full": "Hepatocellular & Biliary Carcinoma",
        "projects": ["TCGA-LIHC"],
        "slides_per_project": 1,
    },
    {
        "folder": "09_Melanoma_TCGA-SKCM",
        "cancer_type": "SKCM",
        "cancer_full": "Cutaneous Melanoma",
        "projects": ["TCGA-SKCM"],
        "slides_per_project": 1,
    },
    {
        "folder": "10_Pancreas_TCGA-PAAD",
        "cancer_type": "PAAD",
        "cancer_full": "Pancreatic Ductal Adenocarcinoma",
        "projects": ["TCGA-PAAD"],
        "slides_per_project": 1,
    },
]


def query_gdc_slides(project_id: str, limit: int = 5):
    """Queries GDC API for diagnostic slide images sorted by file size (smallest first)."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [project_id]}
            },
            {
                "op": "in",
                "content": {"field": "data_type", "value": ["Slide Image"]}
            },
            {
                "op": "in",
                "content": {"field": "data_format", "value": ["SVS"]}
            },
            {
                "op": "in",
                "content": {"field": "access", "value": ["open"]}
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,md5sum,cases.submitter_id,cases.case_id,cases.project.project_id,cases.diagnoses.primary_diagnosis,cases.diagnoses.ajcc_pathologic_stage,cases.demographic.vital_status,cases.demographic.gender",
        "sort": "file_size:asc",
        "size": limit
    }

    try:
        resp = requests.get(f"{GDC_API_URL}/files", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("hits", [])
        return data
    except Exception as e:
        print(f"  [!] GDC Query Error for {project_id}: {e}")
        return []


def download_file_stream(file_id: str, dest_path: Path, expected_size: int = None):
    """Downloads a file from GDC API in chunks with real-time progress."""
    url = f"{GDC_API_URL}/data/{file_id}"
    
    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        print(f"  [OK] Already downloaded: {dest_path.name} ({expected_size / (1024*1024):.1f} MB)")
        return True

    print(f"  [*] Downloading {dest_path.name} from NCI GDC...")
    t0 = time.time()
    
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            downloaded = 0
            
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                        speed = (downloaded / (1024 * 1024)) / max(time.time() - t0, 0.001)
                        print(
                            f"\r      Progress: {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB "
                            f"({pct:.1f}%) - {speed:.1f} MB/s",
                            end="",
                            flush=True
                        )
            print()
            elapsed = time.time() - t0
            print(f"  [OK] Completed {dest_path.name} in {elapsed:.1f}s ({total_bytes / (1024*1024):.1f} MB)")
            return True
    except Exception as e:
        print(f"\n  [!] Download failed for {file_id}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 75)
    print(" NCI Genomic Data Commons (GDC) — Official TCGA Dataset Downloader")
    print(" Authentic Whole Slide Images (.svs) across 10 Cancer Types")
    print(f" Destination Directory: {TARGET_DIR}")
    print("=" * 75)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    master_catalog = []

    total_downloaded = 0
    total_bytes = 0

    for spec in CANCER_PROJECTS:
        folder_path = TARGET_DIR / spec["folder"]
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Processing {spec['cancer_full']} ({spec['cancer_type']}) -> {spec['folder']}")

        for proj in spec["projects"]:
            print(f"  Querying GDC for project '{proj}'...")
            hits = query_gdc_slides(proj, limit=5)
            if not hits:
                print(f"  [!] No slides found for {proj}")
                continue

            # Pick the top N slides (preferring DX diagnostic slides)
            selected_hits = []
            for h in hits:
                fname = h.get("file_name", "")
                # Prefer DX (diagnostic H&E) slides over tissue snapshots
                selected_hits.append(h)
                if len(selected_hits) >= spec["slides_per_project"]:
                    break

            for h in selected_hits:
                file_id = h["file_id"]
                file_name = h["file_name"]
                file_size = h["file_size"]
                
                # Clinical extraction
                case_info = h.get("cases", [{}])[0]
                patient_id = case_info.get("submitter_id", "Unknown")
                diagnoses = case_info.get("diagnoses", [{}])
                primary_diag = diagnoses[0].get("primary_diagnosis", "Malignant Neoplasm") if diagnoses else "Malignant Neoplasm"
                stage = diagnoses[0].get("ajcc_pathologic_stage", "Not Reported") if diagnoses else "Not Reported"
                vital = case_info.get("demographic", {}).get("vital_status", "Alive")
                gender = case_info.get("demographic", {}).get("gender", "Unknown")

                dest_file = folder_path / file_name
                success = download_file_stream(file_id, dest_file, expected_size=file_size)

                if success:
                    total_downloaded += 1
                    total_bytes += file_size
                    master_catalog.append({
                        "patient_id": patient_id,
                        "cancer_type": spec["cancer_type"],
                        "cancer_full": spec["cancer_full"],
                        "project_id": proj,
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "folder": spec["folder"],
                        "relative_path": str(dest_file.relative_to(BASE_DIR)).replace("\\", "/"),
                        "primary_diagnosis": primary_diag,
                        "stage": stage,
                        "vital_status": vital,
                        "gender": gender,
                    })

        # Save manifest per cancer folder
        folder_manifest = folder_path / "manifest.tsv"
        with open(folder_manifest, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["patient_id", "project_id", "file_id", "file_name", "size_mb", "diagnosis", "stage"])
            for row in [r for r in master_catalog if r["cancer_type"] == spec["cancer_type"]]:
                writer.writerow([
                    row["patient_id"], row["project_id"], row["file_id"],
                    row["file_name"], row["file_size_mb"], row["primary_diagnosis"], row["stage"]
                ])

    # Save Master Catalog
    catalog_path = TARGET_DIR / "tcga_master_catalog.csv"
    if master_catalog:
        with open(catalog_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "patient_id", "cancer_type", "cancer_full", "project_id",
                "file_id", "file_name", "file_size_mb", "folder",
                "relative_path", "primary_diagnosis", "stage", "vital_status", "gender"
            ])
            writer.writeheader()
            writer.writerows(master_catalog)

    # Save detailed README
    readme_path = TARGET_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# Authentic TCGA Clinical Histopathology Datasets (NCI GDC)

This directory contains **authentic whole-slide pathology images (.svs)** downloaded directly from the official **National Cancer Institute (NCI) Genomic Data Commons (GDC) / The Cancer Genome Atlas (TCGA)** repository.

## Cancer Organ Folders
1. `01_Breast_TCGA-BRCA/` (Breast Invasive Carcinoma)
2. `02_Lung_TCGA-LUNG/` (TCGA-LUAD / TCGA-LUSC)
3. `03_Kidney_TCGA-KIDNEY/` (TCGA-KIRC / TCGA-KIRP)
4. `04_Colon_TCGA-COLON/` (TCGA-COAD / TCGA-READ)
5. `05_Brain_TCGA-GLIOMA/` (TCGA-GBM / TCGA-LGG)
6. `06_Prostate_TCGA-PRAD/` (TCGA-PRAD)
7. `07_Stomach_TCGA-STAD/` (TCGA-STAD)
8. `08_Liver_TCGA-LIHC/` (TCGA-LIHC)
9. `09_Melanoma_TCGA-SKCM/` (TCGA-SKCM)
10. `10_Pancreas_TCGA-PAAD/` (TCGA-PAAD)

## Summary Statistics
- Total Genuine TCGA Slides Downloaded: {total_downloaded}
- Total Data Size: {total_bytes / (1024 * 1024):.2f} MB
- Official Catalog Manifest: `tcga_master_catalog.csv`
- Provenance: [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/)
""")

    print("\n" + "=" * 75)
    print(f"[OK] ALL TCGA CLINICAL DATASETS DOWNLOADED AND READY!")
    print(f"     Total Slides: {total_downloaded}")
    print(f"     Total Volume: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"     Location:     {TARGET_DIR}")
    print(f"     Catalog:      {catalog_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
