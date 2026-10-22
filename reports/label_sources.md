# Data Sources and Label Mapping Documentation

## 1. GDC Clinical API
- **Endpoint**: `https://api.gdc.cancer.gov/cases`
- **Target Projects**: TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-KIRC, TCGA-KIRP, TCGA-KICH, TCGA-COAD, TCGA-READ, TCGA-GBM, TCGA-LGG
- **Columns retrieved**:
  - `submitter_id` -> 12-char Patient Barcode (`patient_id`)
  - `project.project_id` -> Project identifier and primary cancer type
  - `diagnoses.primary_diagnosis` -> Histologic type (IDC: 'Infiltrating duct carcinoma, NOS', ILC: 'Lobular carcinoma, NOS')
  - `diagnoses.ajcc_pathologic_stage` -> Pathological TNM stage
  - `demographic.age_at_index` -> Patient age at diagnosis
  - `demographic.gender` -> Patient sex
  - `demographic.vital_status` & `days_to_death` / `days_to_last_follow_up` -> Overall Survival (OS_time, OS_event)

## 2. cBioPortal DataHub Clinical & Subtype Files
- **BRCA PAM50 Subtypes**: `brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt`
  - Column `SUBTYPE`: BRCA_LumA, BRCA_LumB, BRCA_Her2, BRCA_Basal, BRCA_Normal
  - Filter: Mapped to LumA, LumB, HER2, Basal. Normal-like (N=36) evaluated under counting rule.

- **COADREAD MSI Subtypes**: `coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt` & `data_clinical_sample.txt`
  - Columns: `SUBTYPE`, `MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`
  - Mapping: COAD_MSI / READ_MSI / MANTIS > 0.4 / MSIsensor >= 10 -> `MSI-H`. CIN / GS / POLE / MANTIS <= 0.4 -> `MSS`.

- **Glioma IDH Status**: `lgggbm_tcga_pub/data_clinical_sample.txt` (Ceccarelli et al.)
  - Column: `IDH_STATUS` ('WT' -> `IDH-wt`, 'Mutant' -> `IDH-mut`)

