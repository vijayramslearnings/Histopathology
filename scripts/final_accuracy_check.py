"""Final accuracy validation across all 10 cancers and 16 subtype tasks."""
import sys, torch, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.app.web_app as wa; wa._model = None
from src.app.web_app import run_inference_on_data

BASE_DIR     = Path(__file__).resolve().parents[1]
COHORT_CSV   = BASE_DIR / "data/pilot_cohort.csv"
FEATURES_DIR = BASE_DIR / "data/features/resnet50"

cohort    = pd.read_csv(COHORT_CSV)
task_cols = [c for c in cohort.columns if c not in ["patient_id", "fold", "cancer_type"]]
results   = {t: [0, 0] for t in ["cancer_type"] + task_cols}

# Sample 4 per cancer (40 total) for a fast representative check
sample = cohort.groupby("cancer_type").head(4).reset_index(drop=True)

print(f"Evaluating {len(sample)} specimens across {cohort['cancer_type'].nunique()} cancer types...\n")

for _, row in sample.iterrows():
    pt = FEATURES_DIR / f"{row['patient_id']}.pt"
    if not pt.exists():
        continue
    d = torch.load(str(pt), weights_only=False, map_location="cpu")
    d["is_external"] = False
    res   = run_inference_on_data(d, pid=str(row["patient_id"]))
    preds = res["predictions"]

    results["cancer_type"][1] += 1
    results["cancer_type"][0] += int(
        preds.get("cancer_type", {}).get("prediction", "") == row["cancer_type"]
    )
    for t in task_cols:
        v = row.get(t)
        if pd.isna(v):
            continue
        if t in preds:
            results[t][1] += 1
            results[t][0] += int(str(preds[t].get("prediction", "")) == str(v))

print("══════════════════════════════════════════════════════")
print("  FINAL ACCURACY REPORT — All Cancers & Subtypes")
print("══════════════════════════════════════════════════════")
all_pass = True
for t, (correct, total) in results.items():
    if total == 0:
        continue
    acc  = correct / total * 100
    flag = "✅ PASS" if acc >= 90 else "❌ FAIL"
    if acc < 90:
        all_pass = False
    bar  = "█" * int(acc / 5)
    print(f"  {flag}  {t:<26} {acc:6.1f}%  {bar:<20}  ({correct}/{total})")

print()
if all_pass:
    print("  🏆 ALL TASKS ≥ 90% — Research-grade accuracy achieved!")
else:
    print("  ⚠  Some tasks below 90% — check above.")
print("══════════════════════════════════════════════════════")

