# Fire Early Warning
## Faculty viva study guide | 8 October 2026

**Core message:** We built a location-based demonstration and investigated why strong sampled wildfire benchmarks do not automatically produce useful real-world alerts.

### 1. Objective and workflow
Explore pre-event wildfire risk from environmental conditions. The current interface accepts a location and shows an XGBoost model score, environmental context, a map and data-quality warnings. It is a research prototype for a Greece-centred domain.

**Location -> environmental data -> fixed feature pipeline -> saved XGBoost -> wildfire-risk score -> map and explanation.**

The label day is t. The principal forecast uses only the preceding ten days, t-10 through t-1. Fire-outcome layers provide labels, not predictors. Recognising visible fire in an image is a separate supporting MVP, distinct from forecasting future risk.

### 2. Three categories to keep separate
**Established results:** Exp1-5, the Exp5 artifact recovery and A/B/C failure analysis, and Exp7 have saved evidence. Sampling, event rarity, spatial scale and alert budgets strongly affect apparent performance.

**Current prototype:** Streamlit location search, supported-coordinate checks, historical/recent environmental inputs, Track-B XGBoost inference, map display and missing/proxy-data reporting. The separate Historical FireCube tab uses the recovered Exp5 model.

**Proposed work:** Source-aligned data, geographic validation, regional adaptation, global coverage, optional FIRMS/context/news and reviewed alerts. Exp8 is a proposal, not a completed result.

### How to revise
Pages 2-5 cover data, experiments and models. Pages 6-7 cover architecture, limitations and viva answers. Pages 8-9 contain speaking notes. Page 10 is the one-page revision sheet.

**Evidence rule:** Saved metrics and later reports take precedence over older planning notes. Archived RESEARCH_STATUS.md contains outdated statements that B/C were unexecuted, plus a later correction. This guide follows the completed A/B/C and Exp7 reports.

**Terminology:** This project's "Track-B production" name means its separate app baseline. It is not Mesogeos Track B, which concerns final burned area after ignition. "Production" names an artifact/interface; it does not establish operational readiness.

**Scope of this guide:** Repository inspection and documentation only. No training, experiments or model changes were performed. Source references S1-S9 are indexed on page 9.

---PAGE---
# Datasets and evaluation
### 3. Data actually used [S1-S3, S8-S9]
**FireCube:** Local Greece-centred NetCDF covering 2009-2021, nominally kilometre-scale daily cells. File size: 23,501,672,194 bytes. It combines weather, vegetation, surface temperature, soil moisture, terrain, land cover, population/infrastructure and fire labels. FWI provides a weather-index baseline under the same protocol.

**Why FireCube?** Aligned environmental history, labels, an extreme-year test and an accessible local cube made it feasible on this laptop. It is not global. The research label is the dataset's fire-start-day burned-cell target, not a record of every possible ignition.

**Extracted samples:** 78,960 examples with a 16-day history buffer; H=1 uses its last ten days. Train: 53,208; validation: 5,469; test 2020: 4,791; test 2021: 15,492. Easy negatives use two negatives per positive, giving sampled prevalence of 1/3. The archive retains the samples.

**Algerian Forest Fires:** Legacy Random Forest uses 244 cleaned rows from two regions, ten weather/FWI inputs and a stratified 195/49 train/test split. It supports the manual Weather MVP, not the main Track-B model.

**Images:** Available training/validation folders contain synthetic fire/no-fire images: 160 training and 40 validation examples, matching saved metric counts. They demonstrate the image pipeline; they do not establish field performance. Real datasets suggested in the README are not proof of use.

**Live sources:** Nominatim resolves place names. Open-Meteo supplies modelled weather and a soil-moisture proxy. ORNL MODIS adapters supply available NDVI and daytime/nighttime surface-temperature composites. Static geography still comes from historical FireCube. APIs can be unavailable.

**Mesogeos:** Reviewed as a wider Mediterranean benchmark; reference code is archived. No completed local Mesogeos-trained experiment is evidenced. It was not the training source for the reported local results.

### 4. Evaluation terms
**Chronological split:** FireCube research trains on 2009-2018, validates on 2019 and reports 2020/2021 separately. Full-grid studies mainly cover May-September; 2021 has 121 evaluable days, not an assumed full 153-day season.

**Precision:** Correct alerts divided by all alerts. **Recall:** Detected positives divided by all actual positives. **ROC-AUC:** Ability to rank positives above negatives across thresholds. **Average precision (AP):** Precision-recall ranking summary; reports call this PR-AUC and use average_precision_score. AP depends on prevalence: sampled and full-grid AP describe different evaluation settings.

**Brier score:** Mean squared probability error. **Calibration:** Whether stated probabilities match observed frequencies. Low Brier under extreme rarity alone does not establish useful alerts. **Area budget:** How much land can be prioritised for review; connects model ranking to a practical decision.

---PAGE---
# Completed research experiments
### 5. Experiment ledger [S2-S5]
**Exp1 - Easy-negative 1 km benchmark: completed.** Compared FWI, XGBoost and a 32-hidden-unit LSTM with matched inputs/splits. Sampled 2020 XGBoost AP=0.9050 and ROC=0.9375; LSTM AP=0.8696; FWI AP=0.8285. XGBoost led this comparison. These are sampled classification results, not operational accuracy.

**Exp1 lead-time check: completed.** Existing models were evaluated with shifted ten-day windows ending 1, 3, 5 or 7 days before t. Sampled 2020 XGBoost AP falls from about 0.905 at H=1 to about 0.87 at H=7; LSTM falls from about 0.870 to 0.80. This is input-window sensitivity, not separately established operational forecasts at every horizon.

**Exp2 - Hard-negative/full-grid evaluation: completed.** Tested the easy-training approach under harder conditions. Sampled-hard 2020 XGBoost AP=0.469, ROC=0.574. Full-grid 2020 ROC remains about 0.869, but AP is only 0.000147 with prevalence about 0.0000175. Easy-negative success does not imply useful precision across a landscape.

**Exp3 - Hard-negative training at 1 km: completed.** Sampled-hard 2020 AP improves to about 0.769, but full-grid AP is about 0.0000372 and ROC=0.707, worse than Exp2. Harder training examples did not solve the full-grid decision problem.

**Exp4 - Spatial pooling: completed.** Aggregated existing 1 km predictions into approximately 5 km, 10 km and NUTS-3 units without retraining. Approximately 10 km max pooling was the preferred practical pooled scale among those studied. Absolute precision remained low. This is spatial aggregation, not a held-out-country generalisation test.

**Exp5 - Direct approximately 10 km XGBoost: completed.** Directly trained unit-day features over 4,022 units, with full validation/test grids; training subsampled negatives at 20:1. In 2020 AP=0.007295 and ROC=0.8743; in 2021 AP=0.006233 and ROC=0.8444. Direct training improved 2020 AP over Exp4 easy-model pooling (~0.00297) and pooled FWI (~0.00080). Practical precision stayed low.

**Exp5 artifact recovery: completed.** Reran the same methodology to retain feature/prediction/model exports for analysis. The later analysis reports exact recomputation agreement for saved 2020/2021 AP and ROC. This is reproducibility/artifact recovery, not a new scientific hypothesis.

**Exp6 - Hard-negative training at 10 km: not completed.** Prepared scripts/configuration and a blocked-launch report exist; no completed metrics/model establish an outcome. Do not describe its proposed ML method as successful or unsuccessful.

---PAGE---
# Failure analysis and the latest experiment
### 6. Exp5 failure analysis A/B/C: completed [S4]
**A - Precision at useful recall.** With the observed Exp5 scores, threshold tuning cannot reach 5% precision at 20-30% recall in either year. At approximately 20% recall, precision is 1.36% in 2020 and 1.16% in 2021. This is an empirical constraint on these scores, not a universal ceiling for future models.

**B - Within-fire-day spatial ranking.** Mean per-day XGBoost ROC=0.8111 (2020) and 0.7726 (2021), versus FWI=0.5401/0.5347. Land-cover-matched XGBoost ROC is about 0.767 in both years. Local ranking contains signal; the hypothesis of near-random same-day separation was not supported.

**C - Domain-day ranking.** Domain-mean FWI separates fire days from non-fire days with ROC=0.8423/0.8603; domain-mean XGBoost gives 0.8369/0.8237. Both day-level and local information matter. These differently aggregated tasks are not a formal causal decomposition.

**Why low precision despite signal?** In 2020, 206 positives occur among 615,366 evaluated unit-days, about 0.0335%. At around 20% recall, approximately 42 hits occur among 3,082 flagged unit-days. High lift over a tiny base rate still permits many false alarms. The analysis does not prove that better features or architectures could never help.

### 7. Exp7 - Two-stage conditional XGBoost: completed [S5]
Stage A predicts fire somewhere in the domain on a day. Stage B scores units conditional on a fire day. Their product is compared with flat Exp5. Stage A has 66 aggregated features; Stage B has 35. Platt calibration for A and isotonic calibration for B use 2019 only. An isotonic-calibrated flat baseline gives a fairer probability comparison.

**Improvement:** Two-stage AP=0.009105 (2020) and 0.006695 (2021), compared with flat raw AP=0.007295/0.006233. Recall improved at the two tightest global area budgets: 500 and 1,000 square kilometres per day.

**Main verdict:** It failed the predefined success rule: AP improvement plus at least 3/5 budget-recall improvements in both years. Only 2/5 improved each year. Wider budgets tied or worsened; calibration was broadly similar to calibrated flat Exp5.

**Mechanism:** Stage A is constant across units on a day, so multiplication cannot improve that day's ordering. It can redistribute attention between days. Stage B's within-day ROC was lower than flat Exp5: 0.7752/0.7658 versus 0.8111/0.7726.

**Exp8 - Proposed only:** Combine Stage A with the better flat Exp5 within-day ranker. No completed Exp8 evidence was found. A report's recommendation is not a result.

---PAGE---
# Models and the current app baseline
### 8. Important models and why they were used
**FWI:** Weather-based danger index and interpretable reference, not a trained neural network. Excluded from the ten FireCube ML dynamic inputs; included in the legacy Algerian Random Forest inputs. Exp7 Stage A includes aggregated FWI.

**XGBoost:** Boosted trees capture nonlinear environmental interactions at modest inference cost. Main research baseline and separate app model. Exp1 supports its choice over the tested LSTM within that setup, not universal superiority of trees.

**LSTM:** Recurrent network tested in Exp1 to model ten-day sequences directly. It did not surpass XGBoost on the reported sampled AP comparison. Not the main app model.

**Random Forest:** Separate Algerian weather MVP. Saved test accuracy=0.9796, F1=0.9825 on only 49 test rows. These metrics do not describe FireCube or the live prototype.

**MobileNetV2:** ImageNet-initialised CNN adapted to two classes; frozen-backbone training then fine-tuning. Saved validation accuracy=1.0 on 40 available synthetic examples. This supports demonstration, not field detection claims.

**Late fusion:** Legacy weighted image/weather score; default image weight=0.6. No independent fused-system evaluation was found. Not combined with Track-B.

**Platt/isotonic calibration:** Score mappings used in Exp7/historical research. They do not ensure calibrated probabilities after geographic, prevalence or provider shifts. ConvLSTM/Transformer/global alternatives appear in literature/plans, not completed local model results.

### 9. Current Track-B production baseline [S6, S9]
Artifact: models/production/xgb_baseline.joblib. Distinct from Exp5 and Exp7. Training specifies 250 trees, depth 6, learning rate 0.05, row/column subsampling 0.8 and seed 42.

**35 inputs = 20 dynamic summaries + 5 static + 10 land-cover fractions.** Dynamic variables: air temperature, dew point, relative humidity, rainfall, surface pressure, wind, NDVI, day/night land-surface temperature and soil-moisture index. Each contributes a ten-day mean and t-1 value. Static inputs: elevation, slope, road distance, waterway distance and population. CLC supplies ten fractions.

Median imputation and StandardScaler are fit on training data only and saved. Scaling preserves the saved preprocessing contract; trees do not inherently need standardisation. Feature names/order are checked at model load.

**Recorded validation:** 53,208 train; 5,469 validation samples from 2019; 1/3 positives in both. Validation ROC=0.9202, AP=0.8553. No production full-grid/global test metric is recorded here. operational_validity is explicitly false.

**Critical mismatch:** Training uses sampled cell features, whereas serving mixes unit-aggregated static/historical inputs and live point/provider estimates. Matching 35 feature names is not proof of source/scale equivalence. The 0-100 output is a model score, not a calibrated real-world fire percentage.

---PAGE---
# Workflow, architecture and important files
### 10. Demonstration sequence [S9]
1. Launch run_fire_warning.bat to open port 8502. It uses this laptop's configured Python runtime and .venv site-packages.
2. Fire Early Warning defaults to historical mode. Use 38.0 N, 23.7 E, 2020-07-15 for repeatable local input. This runs Track-B with historical features.
3. Switch to Current live-data prototype, search Athens, choose a returned location and predict. Search can return global places; static geography rejects unsupported points.
4. Explain score, modelled-weather cards, map, feature provenance and proxy/missing warnings. Current weather is separate context; features use ten completed UTC days before the target.
5. Research & results provides context. The separate Historical FireCube tab uses recovered Exp5 and its historical calibration. Image, Weather and Combined tabs remain separate MVPs.

### 11. Architecture map
**app.py:** UI, input/mode selection, cached services, maps and explanations.
**src/place_search.py:** Nominatim searches, caching, request spacing and errors.
**src/firecube_live.py:** Open-Meteo/ORNL providers, historical dynamic data, JSON cache and a legacy live adapter. Providers return values with availability/approximation metadata.
**src/production_features.py:** Geography/static provider, ten-day feature construction, saved imputation/scaling and 35-feature schema.
**src/static_geography.py:** Checksum-verified compact Greece mapping/static features for live Track-B; historical replay retains the cube-backed provider. See STATIC_GEOGRAPHY.md for provenance and parity evidence.
**src/production_model.py:** Model/schema checks and prediction service. Withholds live output when all six weather-history means are unavailable.
**src/firecube_inference.py:** Historical Exp5 loading, domain mapping and prediction.
**src/benchmark_direct_10km.py:** Shared aggregation/feature functions imported at runtime despite its research name.
**src/predict.py:** Legacy image/weather/fusion inference.

**models/production/:** Track-B model, preprocessor, schema, metrics and manifest. Other models/ files contain legacy MVP artifacts.
**models/production/greece_static_v1.npz + .json:** Required compact geography for live Track-B. **data/raw/firecube/dataset_greece.nc:** Optional local cube required for historical replay. **data/images/samples/:** UI image examples.
**results/:** Retained runtime research dependencies: recovered Exp5 model/config and Exp7 flat calibrator.
**research_archive/:** Completed experiments, prediction exports, reports, training source/data. **README.md, requirements.txt, CLEANUP_MANIFEST.md:** Run notes, dependencies and cleanup record.

### 12. What tests demonstrate
Tests cover historical feature order/dimension, domain rejection, provider approximation flags, cache behaviour, search errors and weather failures. Prior cleanup checks verified compilation, startup, search/weather retrieval, prediction and map rendering. Software checks are not ML validation.

The pending production-pipeline test correction uses as_dict() to access raw_feature_values. Compact-artifact tests additionally cover integrity, mapping, missing values, inference without NetCDF access and optional cube parity. Actual test counts belong to each verification run. Archived scripts retain original relative paths/imports; path restoration may be needed before research reruns.

---PAGE---
# Scientific limitations and viva answers
### 13. Limitations
Greece temporal holdouts do not establish geographic/global generalisation. Labels do not cover every ignition/cause. Spatial aggregation changes the target; it is not precise ignition-location prediction. Static rasters are historical: population is clamped to available years, CLC is 2018. Static timestamp availability needs auditing before a strict historical real-time-replay claim.

Live weather differs from cube inputs; shallow soil moisture is not the EDO index. MODIS composites can be delayed/reused across days. Imputation enables computation without adding observations. Excluding target-day dynamics addresses one leakage route, not every bias. Global feature importance is neither causal nor a per-request explanation.

### 14. Proposed global-model direction
Define target, region, resolution, horizon and decision budget first. Harmonise global sources, timestamps, units, labels and training/serving transformations. Compare pooled XGBoost/FWI with regional models/adaptation using held-out years AND regions. Measure AP, precision/recall, calibration and area-budget outcomes under realistic prevalence. Validate prospectively before alerts. This is a roadmap, not an implemented or trained global system.

Mesogeos could extend Mediterranean validation, not prove global coverage. FIRMS could later add observed active-fire context; detection is different from forecasting, and target-day detections must not leak into inputs. News/context and notifications are future extensions. Exp8 is a separate, nearer research proposal.

### 15. Likely faculty questions and accurate answers
**Main contribution?** A transparent location-based prototype and experiments showing how sampling, rarity, scale and budgets affect useful wildfire forecasting.

**Why XGBoost?** It models tabular interactions efficiently and beat the tested LSTM on Exp1 sampled AP. That supports this baseline within the tested scope.

**What does 70/100 mean?** The model emitted 0.70; it is not evidence of a 70% real-world fire chance.

**Why high ROC but low precision?** Many negatives are ranked correctly, but a small false-positive rate still creates many false alerts when positives are rare.

**Did two-stage modelling solve it?** No. Exp7 improved AP/tight budgets but failed its overall rule. Stage A cannot change same-day unit ordering.

**How is leakage controlled?** Pre-target dynamics, training-only preprocessing and chronological splits; outcomes are labels only. Source/static timing still needs further audit.

**Will it work in India?** Search can find India, but current static inputs/evaluation are Greece-centred. Indian/global claims need aligned data and regional tests.

**Retrained on each click?** No, saved artifacts perform inference. **API failure?** Availability is reported, missing values are flagged/imputed, and all-missing weather-history means cause the live result to be withheld.

---PAGE---
# Safe claims and short explanations
### 16. Claims we should NOT make
- "98% wildfire forecast accuracy": that small legacy RF result is not the main app's accuracy.
- "100% accurate fire detector": available image evidence is synthetic validation.
- "Calibrated global fire probability": Track-B lacks that validation and geographic coverage.
- "Exp5 metrics measure the production model": these are different artifacts/protocols.
- "Exp6 or Exp8 completed": no completed results establish either claim.
- "Parts B/C were never run": later completed analysis supersedes old notes.
- "Exp7 succeeded overall": its predefined success criterion failed.
- "New features or architectures cannot help": no universal impossibility was proved.
- "Alerts, FIRMS or news already integrated": these remain proposed.
- "All tests pass and archived scripts run unchanged": documented test/path caveats remain.

### 17. 30-second explanation
Our project is a Fire Early Warning prototype that turns a location into environmental features and an XGBoost risk score, then shows a map and data-quality warnings. We also studied historical FireCube data and found that strong results on sampled examples can become low-precision alerts on a full landscape. The current demonstration is Greece-centred. Its main strengths are a working workflow and honest evaluation; it is not yet a validated operational alert system.

### 18. 2-minute explanation
Our objective is to explore wildfire risk before the target day using environmental conditions. The main Streamlit workflow starts with a location. It collects environmental context, builds a fixed feature vector, loads a saved XGBoost model and displays the model score, map and data-quality explanation. Historical mode provides a repeatable local demonstration.

The research uses the Greece-centred FireCube dataset. We use the ten days before the target day, separate training years, validation year 2019 and test years 2020 and 2021. Experiments compare a weather index, XGBoost and LSTM, then examine harder negatives and full-grid evaluation.

The strongest lesson is that evaluation design matters. XGBoost did well on easy sampled examples, but actual positive unit-days are rare. Useful ranking can still generate many false alerts. Direct training on approximately ten-kilometre units improved precision-recall performance but did not produce high precision at useful recall. Two-stage modelling later improved tight-budget results but failed its overall success criterion.

The app uses a separate Track-B XGBoost baseline with 35 features. Its saved metrics describe sampled validation, so I do not present the output as a calibrated real-world fire probability. Live providers introduce proxy and freshness differences. Future work is to align sources, validate across regions and years, compare regional adaptation and only then consider operational alerts.

---PAGE---
# Five-minute explanation and evidence
### 19. Five-minute explanation - timed speaking notes
**0:00-0:45 | Problem.** Explain forecasting versus visible-fire recognition. Introduce location -> environmental data -> features -> XGBoost -> score/map/warnings. Describe image/weather tabs as separate earlier MVPs. State the Greece-centred scope and demonstration status.

**0:45-1:30 | Data/design.** FireCube provides aligned history, labels and FWI in one feasible local cube. Explain chronological splits and t-10..t-1. Give the feature arithmetic: ten dynamic variables summarised twice, five static values and ten land-cover fractions. Describe complementary weather, fuel, terrain and human-access context without claiming causality.

**1:30-2:20 | Experiments 1-4.** Exp1 favoured XGBoost on easy sampled AP. Exp2 revealed that high ROC need not imply precision on hard/full-grid evaluation. Exp3 improved sampled-hard tests while worsening full-grid results. Exp4 examined pooling and preferred approximately 10 km among tested scales. Aggregation is not geographic transfer validation.

**2:20-3:10 | Exp5 and diagnosis.** Direct unit training improved over pooling. Quote 2020 AP ~0.0073 and precision ~1.36% at 20% recall. A/B/C found both day-level and within-day signal. Rare positives still generated many false alerts. Threshold adjustment cannot overcome the observed score curve, but future data/models are not ruled out.

**3:10-3:55 | Exp7/status.** Explain day-level Stage A times conditional-unit Stage B. AP/tight budgets improved, but the overall criterion failed. A day-constant multiplier cannot reorder units within that day. Exp6 has no completed result; Exp8 is a proposed hybrid.

**3:55-4:35 | Show app.** Run 38.0, 23.7, 2020-07-15, then live search if providers are available. Distinguish current weather from the ten-day model history and Track-B from Exp5. Show proxy/missingness/provenance warnings. A map pin marks the query location, not a predicted fire perimeter.

**4:35-5:00 | Conclusion.** Contribution: working transparent prototype and careful rare-event evaluation. Next: aligned sources, held-out regions, realistic prevalence, calibration and prospective validation. Global adaptation, context and alerts remain proposed. Invite questions about catching positives versus false-alert/area-review cost. Timing assumes explanation and brief on-screen demonstration, not reading only the headings.

### Evidence index - paths relative to project root
**S1:** research_archive/research_documentation/docs/DATASET_MESOGEOS_VS_FIRECUBE.md and RESEARCH_EARLY_WARNING.md. Dataset rationale and proposed directions; external literature is not a local result.
**S2:** research_archive/results/ subfolders firecube_benchmark, firecube_hardneg, firecube_hardneg_train, firecube_spatial_agg: metrics.json.
**S3:** research_archive/results/firecube_direct_10km/metrics.json; firecube_direct_10km_artifact_rerun/integrity_report.json under the same results archive; research_archive/research_data/processed/firecube_samples.json.
**S4:** research_archive/results/firecube_exp5_failure_analysis/REPORT.md and SUMMARY.md.
**S5:** research_archive/results/firecube_exp7_two_stage/EXP7_REPORT.md and EXP7_SUMMARY.md; firecube_hardneg_10km/EXP6_AUTOLAUNCH_BLOCKED.txt under the same results archive.
**S6:** models/production/metrics.json and manifest.json; research_archive/research_source/train_production_baseline.py.
**S8:** models/image_metrics.json, tabular_metrics.json and archived training scripts/data. **S9:** app.py, src/, tests/, launcher, README.md and CLEANUP_MANIFEST.md. Reviewed on the cover date; no new experiment run.

---PAGE---
# One-page revision cheat sheet
### Say this first
"A location-based wildfire-risk prototype with transparent data quality, supported by research on realistic rare-event evaluation."

### Three boundaries
**RESULTS:** Exp1-5 + recovery + A/B/C + Exp7 completed. **PROTOTYPE:** Separate Track-B XGBoost, Greece-centred inputs, score/map/provenance. **FUTURE:** Global/regional validation, source alignment, Exp8, context and alerts. Exp6 has no completed result.

### Data and model in one breath
FireCube: Greece-centred, 2009-2021. Train 2009-2018; validate 2019; test 2020/2021. Main window t-10..t-1. No target-day fire outcomes as predictors. 35 inputs = 10 dynamic means + 10 last-day values + 5 static + 10 CLC fractions. Saved imputation/scaling -> XGBoost.

### Experiment memory ladder
**1:** Easy sampled results look strong; XGBoost leads tested LSTM on AP.
**2:** Hard/full-grid tests expose low precision.
**3:** Hard training helps sampled tests but hurts full-grid performance.
**4:** Approximately 10 km max pooling is a useful tested scale.
**5:** Direct 10 km training improves AP; alerts remain weak.
**A/B/C:** Local and day-level signal exist; rarity/coarse ranking limit precision.
**7:** Tight-budget gains, overall criterion fails; day multiplier cannot reorder same-day units.

### Four numerical anchors
- Exp1 sampled 2020 XGBoost AP ~0.905; positives=1/3 of the sample.
- Exp5 full-grid 2020 AP ~0.0073; prevalence ~0.0335%; precision ~1.36% at 20% recall.
- Exp7 budget wins=2/5 per year; required at least 3/5 in both years.
- Track-B: 53,208 train / 5,469 validation; sampled 2019 ROC=0.9202, AP=0.8553. Not Exp5 results or live accuracy.

### Demo and files
run_fire_warning.bat -> localhost:8502. Historical example: 38.0, 23.7, 2020-07-15. app.py=UI; place_search=geocoding; firecube_live=providers; production_features=35-feature pipeline; production_model=model/service; firecube_inference=Exp5; predict=legacy MVP. models/production holds Track-B; research_archive holds evidence.

### Accurate answers
**70/100?** Model score, not verified 70% probability. **High ROC/low precision?** Rare positives mean many false alerts. **Worldwide?** Not validated; current static geography restricts scope. **100% image accuracy?** Available synthetic validation only. **Spatial evaluation?** Aggregation/ranking, not geographic holdout. **API values?** Estimates/composites/proxies with missingness. **One-command research reproduction?** Archive paths and a known test mismatch need attention.

### Closing line
"Our strongest result is understanding where the model is useful and where evaluation can mislead us. We need aligned data and realistic regional validation before operational alerts."
