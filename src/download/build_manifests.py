"""
build_manifests.py
Queries the GDC REST API for Diagnostic Slide images (-DX1) across target TCGA projects.
Filters for one slide per patient and outputs:
  - data/manifests/{project}.tsv (gdc-client manifest format: id, filename, md5, size, state)
  - data/manifests/all_slides_metadata.csv (patient_id, project, tss, file_id, file_name, file_size_mb, md5)
"""

import os
import json
import time
import argparse
import pandas as pd
import httpx
from pathlib import Path

TARGET_PROJECTS = [
    "TCGA-BRCA",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-KICH",
    "TCGA-COAD",
    "TCGA-READ",
    "TCGA-GBM",
    "TCGA-LGG",
]

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def fetch_project_slides(project_id: str, client: httpx.Client, page_size: int = 1000):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
        ]
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "state",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.submitter_id"
    ]

    all_hits = []
    from_idx = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": page_size,
            "from": from_idx
        }

        resp = client.get(GDC_FILES_ENDPOINT, params=params, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()["data"]
        hits = data["hits"]
        all_hits.extend(hits)

        total = data["pagination"]["total"]
        from_idx += len(hits)
        if from_idx >= total or not hits:
            break

    return all_hits


def process_hits_for_project(hits, project_id):
    """
    Keep only files containing '-DX1'.
    Keep one slide per patient (first 12 chars of barcode: TCGA-XX-XXXX).
    Extract TSS (characters 6-7, e.g. barcode[5:7]).
    """
    records = []
    seen_patients = set()

    # Sort hits deterministically so picking is reproducible
    hits_sorted = sorted(hits, key=lambda x: (x.get("file_name", ""), x.get("file_id", "")))

    for h in hits_sorted:
        fname = h.get("file_name", "")
        if "-DX1" not in fname.upper():
            continue

        cases = h.get("cases", [])
        if not cases:
            continue
        case = cases[0]
        patient_barcode = case.get("submitter_id", "")
        if not patient_barcode or len(patient_barcode) < 12:
            continue

        patient_id = patient_barcode[:12].upper()
        if patient_id in seen_patients:
            continue
        seen_patients.add(patient_id)

        tss = patient_id[5:7] if len(patient_id) >= 7 else "NA"

        records.append({
            "patient_id": patient_id,
            "project": project_id,
            "tss": tss,
            "file_id": h.get("file_id", ""),
            "file_name": fname,
            "md5": h.get("md5sum", ""),
            "file_size": h.get("file_size", 0),
            "state": h.get("state", "validated"),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Build GDC manifests for TCGA Diagnostic Slides.")
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary_rows = []

    print("=" * 70)
    print("Fetching GDC Diagnostic Slide Manifests (-DX1, 1 slide per patient)")
    print("=" * 70)

    with httpx.Client() as client:
        for proj in TARGET_PROJECTS:
            try:
                hits = fetch_project_slides(proj, client)
                records = process_hits_for_project(hits, proj)
                all_records.extend(records)

                # Write project manifest in gdc-client TSV format
                manifest_path = out_dir / f"{proj}.tsv"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    f.write("id\tfilename\tmd5\tsize\tstate\n")
                    for r in records:
                        f.write(f"{r['file_id']}\t{r['file_name']}\t{r['md5']}\t{r['file_size']}\t{r['state']}\n")

                total_size_gb = sum(r["file_size"] for r in records) / (1024 ** 3)
                summary_rows.append({
                    "Project": proj,
                    "Total GDC Hits": len(hits),
                    "DX1 Selected Slides (Patients)": len(records),
                    "Total Size (GB)": round(total_size_gb, 2)
                })
                print(f"[{proj:10s}] Hits: {len(hits):4d} | DX1 Patients: {len(records):4d} | Size: {total_size_gb:6.2f} GB")
            except Exception as e:
                print(f"Error fetching {proj}: {e}")

    df_meta = pd.DataFrame(all_records)
    meta_path = out_dir / "all_slides_metadata.csv"
    df_meta.to_csv(meta_path, index=False)
    print(f"\nSaved metadata to: {meta_path}")

    df_summary = pd.DataFrame(summary_rows)
    print("\nSummary:")
    print(df_summary.to_string(index=False))
    total_slides = df_summary["DX1 Selected Slides (Patients)"].sum()
    total_gb = df_summary["Total Size (GB)"].sum()
    print("-" * 70)
    print(f"GRAND TOTAL: {total_slides} slides ({total_gb:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
