# Research status — FireCube early-warning track (Exp1–Exp5 + Part A)

Date: 2026-09-14  
Scope: leakage-safe pre-ignition risk (predictors **t−10…t−1** only; never `burned_areas` / `ignition_points` / `number_of_fires` as features).  
Splits: train ≤2018, val 2019, test 2020 & 2021 (May–Sep summers unless noted).  
**Exp6 not run. Parts B/C of the Exp5 failure analysis not executed. No invented B/C results.**

---

## What Exp1–Exp5 established

| Exp | Setup | Main result |
|-----|--------|-------------|
| **1** | 1 km cell-day; easy negatives (~1:2); FWI / XGB / LSTM | On **easy eval**, models look strong (XGB best). |
| **2** | Same Exp1 XGB on **hard negatives** + **full summer grid** | Easy success **does not transfer**. Sampled hard PR collapses; full-grid ROC stays high while PR ≈ prevalence. |
| **3** | Hard-negative **training** at 1 km | Helps **sampled** hard tests; **hurts** full-grid PR/ROC vs Exp2. |
| **4** | Post-hoc spatial pooling of 1 km scores (no retrain); ~5 / ~10 km / NUTS-3 | ~**10 km max-pool** is the best practical pooled scale among those tried; absolute precision still tiny. |
| **5** | **Direct** ~10 km unit-day XGB (4022 units; full val/test grids) | Beats Exp4 pool and FWI on PR-AUC and P@recall; ROC still strong; **precision at useful recall remains ~1%**. |

Protocol constants reused: cell km² ≈ 0.648; ~10 km strides 11×14; unit positive if any fire-start burned cell in unit on day *t*.

---

## Strongest exact evidence

**1. Easy eval ≠ operations (Exp1 → Exp2)**  
- Exp1 XGB test2020 PR-AUC **0.905**, ROC **0.938** (base rate 1/3).  
- Same model, Exp2 hard sampled test2020: PR-AUC **0.469**, ROC **0.574**.  
- Exp2 **full grid** 2020: ROC **0.869**, PR-AUC **1.47×10⁻⁴**, base rate **1.75×10⁻⁵**.

**2. Hard-only training is not the fix (Exp3)**  
- Sampled hard test2020 PR-AUC **0.769** (looks good).  
- Full grid 2020: PR-AUC **3.72×10⁻⁵**, ROC **0.707** — **worse** than Exp2 full-grid XGB.

**3. Scale + direct 10 km help PR, not operational precision (Exp4/5)**  
- Exp5 direct XGB 2020: PR-AUC **0.00730**, ROC **0.874**, base rate **3.35×10⁻⁴**.  
- vs Exp4 easy-pool A: PR-AUC **0.00297** (Δ **+0.00432**).  
- vs FWI pool C: PR-AUC **0.00080** (Δ **+0.00650**).  
- Exp5 2020 P@≈20% recall: **0.0136**; P@≈50% recall: **0.0033**.  
- F1-tuned thr 0.75 (2020): P=**0.02**, R=**0.485%** (essentially unused recall).

**4. Architecture under easy negatives is secondary (Exp1)**  
- LSTM did not overturn XGB on easy eval; later failures are evaluation/prevalence/scale, not “wrong network family.”

---

## What Part A established

Source: `results/firecube_exp5_failure_analysis/` from Exp5 `metrics.json` ranking curve (raw Exp5 `npz`/`joblib` absent; `precision_at_recall` **is** the empirical score-CDF operating curve).

- At **20% and 30% recall**, Exp5 full-grid precision is **≪ 5%** in both 2020 and 2021.  
- **2020:** P≈**1.36%** / **0.74%** (min FPR **0.0049** / **0.014**).  
- **2021:** P≈**1.16%** / **0.68%** (min FPR **0.011** / **0.028**).  
- Lift vs base rate is real (about **×19–41** at 20% recall) but still far below a ≥5% precision bar.  
- **Conclusion of Part A:** with **observed Exp5 ranking skill**, ≥5% precision at 20–30% recall on full 10 km unit-days is **not achievable** by threshold tuning alone.

---

## Parts B/C — now RESOLVED (2026-09-16)

Executed on recovered artifacts (`results/firecube_direct_10km_artifact_rerun/`, PR/ROC reproduce Exp5 exactly). See `results/firecube_exp5_failure_analysis/REPORT.md`.

- **B (within-fire-day):** XGB per-day AUROC **0.811** (2020) / **0.773** (2021); CLC-matched **0.767 / 0.766**; burned-unit median percentile rank **0.904 / 0.821**. FWI(t−1) per-day AUROC only **0.540 / 0.535**.
- **C (domain-day):** AUROC domain-mean FWI **0.842 / 0.860**; domain-mean XGB **0.837 / 0.824**.
- **Verdict:** both levels carry real signal (gap ≈ 0.039). The earlier "same-day spatial twins" hypothesis is **refuted**. Local ranking is the ML model's unique contribution over FWI; the ~1% precision is **prevalence + coarse ranking**, i.e. a **decision-formulation** problem.

---

## Previously unresolved (superseded by the section above)

| ID | Question | Why blocked |
|----|----------|-------------|
| **B** | On **fire days only**, how separable are burned vs unburned 10 km units (CLC-matched) under Exp5 predictors / scores? Within-fire-day AUROC? | Needs unit-day arrays or NC extract + process spawn; Agent Shell cannot spawn (`0xC0000005`). |
| **C** | How much skill is **domain-day** meteorological danger (mean/max FWI) vs **local** unit placement? | Same. |

Do **not** treat script stubs or launchers as results.

---

## Claims we can safely make now

1. Easy-negative benchmarks **overstate** operational skill for this label and cube.  
2. Full-grid / hard evaluation shows **high ROC + tiny PR** — ranking of easy mass, not precise alerts.  
3. Hard-negative **training** (Exp3) is not a substitute for fixing prevalence / decision geometry.  
4. Moving to ~**10 km** and training **directly** (Exp5) improves PR vs 1 km / post-hoc pool / FWI, but **does not** reach useful precision at useful recall.  
5. Under Exp5 scores, **thresholding cannot** deliver ≥5% precision at 20–30% recall (Part A).  
6. Chasing another architecture at the **same** unit-day task is **low scientific priority** given Exp1–5 + Part A.

## Claims we must NOT make yet

1. That failures are **mostly** same-day spatial twins (needs **B**).  
2. That skill is **mostly** day-level weather with little local signal (needs **C**, contrasted with B).  
3. That new features (human ignition, fuel continuity, etc.) **will** or **will not** fix precision (untested).  
4. That Exp6 hard-neg at 10 km would help or hurt (not run).  
5. Any numeric within-fire-day AUROC or domain-day AUROC (B/C not executed).  
6. That the product is operationally deployable as a high-recall unit-day alert system under current predictors.

---

## Are B/C essential for the next decision?

**No — not for the decision already forced by Exp1–5 + Part A.**

That decision: **stop optimizing the current unit-day alert operating point** (same labels, same feature family, another model/hard-neg/threshold) as if ≥5% precision at 20–30% recall were nearby.

**Yes — for choosing the *mechanism* of the next scientific move** (day-level ceiling vs local spatial confusion vs new predictors). B/C remain the cheapest way to split those mechanisms **when a normal Windows process can run**, but they are **analysis**, not a new trained forecaster.

**Move-forward stance:** evidence is **sufficient** to change research objective away from “better XGB at Exp5 task.” The single best *next* step is still diagnostic **B/C** (not Exp6, not a new architecture).

---

## Single next experiment (if more work is required)

**Name:** Complete Exp5 failure analysis **Parts B + C only** (script: `src/analyze_exp5_failure.py`; out: `results/firecube_exp5_failure_analysis/`).  

**Hypothesis:** On days with ≥1 fire, burned vs unburned 10 km units are only weakly separable with Exp5 predictors (within-fire-day AUROC near ~0.5–0.6), while domain-day FWI separates fire vs no-fire days more clearly — i.e. meteorology flags **dangerous days** better than **which unit ignites**.  

**Not:** Exp6, new model class, or retraining under Agent Shell.

---

## Artifact map (authoritative)

- Exp1: `results/firecube_benchmark/metrics.json`  
- Exp2: `results/firecube_hardneg/metrics.json`  
- Exp3: `results/firecube_hardneg_train/metrics.json`  
- Exp4: `results/firecube_spatial_agg/metrics.json`  
- Exp5: `results/firecube_direct_10km/metrics.json` (+ `feature_aggregation.json`; **npz/joblib missing**)  
- Part A: `results/firecube_exp5_failure_analysis/analysis_metrics.json`, `SUMMARY.md`, `tables/precision_ceiling.csv`
