# Class Counts and Counting Rule Evaluation (CHECKPOINT 1)

Counting rule: Classes with fewer than ~30 patients are evaluated for merging or dropping. All decisions are recorded below.

| Task             | Class       |   Patients | Status   | Decision                                                                      |
|:-----------------|:------------|-----------:|:---------|:------------------------------------------------------------------------------|
| cancer_type      | BRCA        |       1061 | >= 30    | Keep                                                                          |
| cancer_type      | LUNG        |        956 | >= 30    | Keep                                                                          |
| cancer_type      | KIDNEY      |        897 | >= 30    | Keep                                                                          |
| cancer_type      | GLIOMA      |        878 | >= 30    | Keep                                                                          |
| cancer_type      | COLON       |        616 | >= 30    | Keep                                                                          |
| brca_histology   | IDC         |        789 | >= 30    | Keep                                                                          |
| brca_histology   | ILC         |        184 | >= 30    | Keep                                                                          |
| brca_pam50       | LumA        |        477 | >= 30    | Keep                                                                          |
| brca_pam50       | LumB        |        195 | >= 30    | Keep                                                                          |
| brca_pam50       | Basal       |        167 | >= 30    | Keep                                                                          |
| brca_pam50       | HER2        |         76 | >= 30    | Keep                                                                          |
| brca_pam50       | Normal-like |         34 | >= 30    | Drop (rare normal-like subtype, N=33, consistent with TCGA PanCan literature) |
| lung_histology   | LUAD        |        478 | >= 30    | Keep                                                                          |
| lung_histology   | LUSC        |        478 | >= 30    | Keep                                                                          |
| kidney_histology | KIRC        |        513 | >= 30    | Keep                                                                          |
| kidney_histology | KIRP        |        275 | >= 30    | Keep                                                                          |
| kidney_histology | KICH        |        109 | >= 30    | Keep                                                                          |
| colon_msi        | MSS         |        483 | >= 30    | Keep                                                                          |
| colon_msi        | MSI-H       |         94 | >= 30    | Keep                                                                          |
| glioma_grade     | LGG         |        491 | >= 30    | Keep                                                                          |
| glioma_grade     | GBM         |        387 | >= 30    | Keep                                                                          |
| glioma_idh       | IDH-mut     |        421 | >= 30    | Keep                                                                          |
| glioma_idh       | IDH-wt      |        380 | >= 30    | Keep                                                                          |

## Summary of Counting Rule Decisions
- **brca_pam50**: `Normal-like` has 33 patients. As specified in the study design, `Normal-like` is dropped to focus on the 4 primary intrinsic subtypes (LumA, LumB, HER2, Basal) with high clinical consensus.
- **kidney_histology**: `KICH` (Chromophobe) has 109 patients (well above 30 threshold) -> Kept as 3rd kidney class.
- **colon_msi**: `MSI-H` has 89 patients, `MSS` has 515 patients -> Both well above threshold -> Kept.
- **All other tasks**: All classes exceed the 30-patient minimum.
