# Forest Fire Early Warning & Risk Prediction — Research Report

**Status:** literature and design phase only. No final model implemented.  
**Scope date:** research surveyed through September 2026.  
**Task:** pre-ignition occurrence / fire-danger forecasting, not active-fire detection or spread.

How to read this document:

- **Source-reported** = numbers or claims taken from a paper.
- **Our inference** = interpretation, not claimed by the original authors.
- Results across papers are **not interchangeable**; datasets, labels, horizons, and splits differ.

---

## 1. Executive summary

The scientifically meaningful product for this project is a **short-horizon probability of wildfire occurrence (or of conditions that allow a fire to ignite and become large)** for a user-specified area, updated as new weather and satellite observations arrive. That is distinct from detecting a fire that is already burning and from predicting how a known fire will spread.

There is **no single architecture that is currently state-of-the-art across settings**. What the literature actually supports:

- **Operational fire-weather indices** (Canadian FWI / EFFIS, NFDRS, FFDI) are the real-world baseline. Machine-learning models that add vegetation, soil moisture, land cover, and human activity **often outperform FWI** when evaluated as occurrence classifiers (Kondylatos et al., 2022; Shmuel et al., 2023, 2025; FIRE-RES D1.6, 2025).
- **Gradient-boosted trees (XGBoost / LightGBM / RF)** remain extremely strong, especially on tabular drivers and for **static susceptibility**. Several 2025 studies find they match or beat CNNs/ConvLSTMs for occurrence, and that **region-specific models beat global ones** (Illarionova et al., 2025; Shmuel et al., 2025).
- **Temporal deep models (LSTM / ConvLSTM)** are the strongest *reproducible* next-day **danger** models on Mediterranean datacubes (FireCube / Mesogeos), with **LSTM ≈ ConvLSTM** on F1; ConvLSTM tends to be more precise, LSTM more sensitive (Kondylatos et al., 2022). Transformers on Mesogeos Track A are **not clearly better** than LSTM (AUPRC ~0.85–0.87 for LSTM, Transformer, and GTN).
- **Vision Transformers** show modest gains on **seasonal burned-area pattern** forecasting at coarse global scale (TeleViT / SeasFire), which is a **different task** (weeks–months, ~25 km) than local early warning.
- **Next Day Wildfire Spread** and **WildfireSpreadTS** are important public benchmarks but they predict **spread of already-detected fire**, not pre-ignition risk.

**Recommended research path (not a committed architecture):** adopt a **Mesogeos-style 1 km daily datacube formulation** (or FireCube if Greece-only), treat **FWI + logistic regression + XGBoost** as mandatory baselines, then test whether a **10-day LSTM (and optionally ConvLSTM)** improves **PR-AUC, calibration, and lead-time at fixed false-alarm rate** under **year holdout and geographic holdout**. Only after that evidence should we consider attention fusion or transformers. The most defensible improvement is not “a bigger network”; it is **fair evaluation + multimodal drivers + temporal context + calibrated probabilities + missing-data robustness + cross-region testing**.

---

## 2. Problem definition

**User-facing problem.** A user defines a geographic area. The system repeatedly ingests environmental and satellite information for that area and estimates whether a **forest fire is likely in the near future**. If probability exceeds a threshold, it maps the area as high-risk, explains drivers, alerts the user, and continues monitoring.

**Scientific problem.** Estimate

\[
p(\text{fire event in spatial unit } s \text{ during horizon } [t, t+H] \mid \text{observations up to } t, \text{ and weather forecasts through } t+H)
\]

where the event definition, \(s\), and \(H\) must be fixed before model comparison.

**Constraints implied by the product.** Continuous operation; missing/cloudy observations; alerts that are useful only if **lead time > 0** and **false alarms are tolerable**; explanations of why risk is high.

---

## 3. Detection vs risk vs occurrence forecasting vs spread

These tasks are routinely conflated. They must stay separate.

| Task | Question | Typical labels | Typical horizon | Primary sensors |
| --- | --- | --- | --- | --- |
| **Active-fire detection** | Is something burning *now*? | Hotspots (MODIS/VIIRS), thermal anomalies | Minutes–hours (observation latency) | VIIRS 375 m, MODIS 1 km, GOES, Sentinel-3 SLSTR |
| **Susceptibility / static risk** | Which places burn *in general*? | Long-term burned pixels vs never-burned | Years (often no explicit forecast) | DEM, land cover, climate normals, fire history |
| **Fire danger / occurrence forecasting** | Will a fire start (or become large) *soon*? | Next-day / next-k-day burned or ignition | Hours–days (sometimes 8-day) | Weather + fuel moisture + vegetation + humans |
| **Spread / behaviour** | Given a fire, where next? | Next-day active-fire mask | 24 h after ignition known | Current fire mask + weather + fuel |

**Source-reported distinction:** Zhao et al. (2024/2025 arXiv 2405.01607) separate detection from risk/danger. Kondylatos et al. (2022) use Pettinari & Chuvieco (2020): danger = conditions that allow ignition *and* spread. Huot et al. (2022) and Gerard et al. (2023) are **spread**. Jain et al. (2020) treat occurrence/susceptibility/risk as one review domain but note they are not identical.

**Our inference:** For *this* project, the primary label is **occurrence/danger forecasting**. Detection is a **secondary overlay** (FIRMS alerts). Spread models are **out of scope** except as negative examples of what not to optimize.

---

## 4. Literature review (focused)

### 4.1 Operational indices (the true baseline)

- **Canadian Forest Fire Weather Index (FWI)** (Van Wagner, 1974): temperature, RH, wind, rain → fuel moisture codes + FWI. Used by **EFFIS** (Europe) and globally.
- **US NFDRS** and **Australian FFDI/McArthur**.
- **Shmuel et al., 2025** (*npj Natural Hazards*): Canadian FWI country-level ROC-AUC **0.69**; American FWI **0.61**; **country-specific decision trees 0.86**; genetic-algorithm-tuned FWI **0.79**. **Source-reported:** a single global tree is much worse. **Inference:** regional fire regimes dominate; “one global SOTA model” is likely the wrong product thesis.
- **Shmuel et al., 2023** (MLFWI, *Mach. Learn.: Sci. Technol.*): XGBoost ROC-AUC **0.98**, PR-AUC **0.94** vs traditional indices 0.62–0.80. **Caution (our inference):** such high ROC on occurrence often indicates easy negatives or spatial leakage; treat as “ML beats FWI on their protocol,” not as operational accuracy.
- **FIRE-RES D1.6 (2025):** probabilistic fire-occurrence models beat FWI on PR curves, but **absolute precision at 50% recall remains low** (e.g. 0.17 vs 0.04 for large fires; much lower for extreme events). **Inference:** beating FWI ≠ operationally quiet alerts.

### 4.2 Reviews

- **Jain et al., 2020** (*Environmental Reviews*): ~300 papers to 2019; RF, MaxEnt, ANN, trees, SVM dominate; occurrence/susceptibility/risk is the largest ML domain; interpretability and process knowledge remain necessary. DOI: [10.1139/er-2020-0019](https://doi.org/10.1139/er-2020-0019)
- **Ghali & Akhloufi, 2023** (*Fire*): DL for detection, mapping, prediction from satellites.
- **Zhao et al., arXiv:2405.01607** (v5): DL + remote sensing for risk; tasks = time series, segmentation, spatiotemporal; collinearity and interpretability emphasized. <https://arxiv.org/abs/2405.01607>
- **MDPI Forests 16(4):704 (2025)** “Advancements in AI Applications for Forest Fire Prediction”: datasets + statistical / classical ML / DL since 2015; calls for multimodal DL, physics-informed models, XAI.
- **Ecological Informatics 2025** survey of the full ML pipeline for wildfire risk (data → deployment). DOI: [10.1016/j.ecoinf.2025.103325](https://doi.org/10.1016/j.ecoinf.2025.103325)

### 4.3 Next-day / short-term danger with datacubes (most relevant)

**Kondylatos, Prapas, et al., 2022.** *Wildfire Danger Prediction and Understanding With Deep Learning.* GRL. DOI: [10.1029/2022GL099368](https://doi.org/10.1029/2022GL099368)

- **Objective:** next-day danger in Greece / E. Mediterranean.
- **Data:** FireCube, 1 km × 1 day, 2009–2021; 25 predictors (weather, vegetation, wetness, humans, static).
- **Target:** probability a cell burns in a **large fire (>30 ha)** next day; inputs strictly \(t'<t\) (meteo at t is a forecast).
- **Models:** RF, XGBoost (instance + 10-day means); LSTM and ConvLSTM on **10-day sequences** + 25×25 km patches for ConvLSTM.
- **Split:** train 2009–2018, val 2019, test **2020 (typical)** and **2021 (extreme)**. Balanced sampling ~2:1 nonfire:fire.
- **Source-reported 2020 F1:** XGBoost 0.782; LSTM 0.804; ConvLSTM 0.806. ConvLSTM precision 0.923 vs LSTM 0.861; LSTM better recall. 2021: LSTM F1 **0.879**, ConvLSTM 0.871.
- **vs FWI:** all ML better AUROC after aggregating to 9 km. DL best.
- **XAI:** wetness (soil moisture, RH) important; last-day wind; 2-day RH; cumulative T and precip.
- **Code/data:** FireCube on Zenodo; notebooks <https://github.com/DeepCube-org/uc3-public-notebooks>
- **Limits:** Mediterranean only; large-fire definition; sampled negatives (not full-grid prevalence); 1 km; next-day only.

**Kondylatos, Prapas, Camps-Valls, Papoutsis, 2023.** Mesogeos, NeurIPS Datasets. arXiv: [2306.05144](https://arxiv.org/abs/2306.05144). Site: <https://orionlab.space.noa.gr/mesogeos/> Code: <https://github.com/Orion-AI-Lab/mesogeos>

- Mediterranean 2006–2022, 1 km × 1 day datacube.
- **Track A (danger):** binary classification; LSTM / Transformer encoder / Gated Transformer; CE weighted by log burned area.
- **Source-reported Track A (paper table):** LSTM F1 0.786 AUPRC 0.853; Transformer F1 0.780 AUPRC 0.856. Leaderboard later lists Transformer AUPRC **0.870**, LSTM **0.856**.
- **Inference:** transformer is **not a decisive win** over LSTM. This is the best public **pre-ignition** benchmark for a regional early-warning analog.

### 4.4 Occurrence forecasting with mixed ML/DL (contradicts “DL always wins”)

**Illarionova et al., 2025.** *Scientific Reports.* [10.1038/s41598-025-94002-4](https://doi.org/10.1038/s41598-025-94002-4)

- **Objective:** occurrence in any of the **next 5 days**, four Russian oblasts, ~650 m, 2012–2022, ~17k verified clustered fires.
- **Models:** RF, XGBoost, Autoencoder, ConvLSTM, Attention MLP, RegNetX.
- **Source-reported:** F1 0.7–0.87 **depending on region**; **need region-specific models**; on balanced tests **RF/XGBoost often advantageous**; CNN/ConvLSTM not uniformly better; feature importance differs by climate zone.
- **Inference:** spatiotemporal DL is not automatically the SOTA for occurrence; **geographic heterogeneity** is a first-order effect.

### 4.5 Global / seasonal (related, coarser)

- **SeasFire cube** (Alonso et al.; Scientific Data 2025): global 0.25°, 8-day, 2001–2021, 50+ variables, burned area from FireCCI/GFED/GWIS. [10.1038/s41597-025-04546-3](https://doi.org/10.1038/s41597-025-04546-3)
- **Prapas et al., TeleViT (ICCVW 2023; journal 2026):** S2S burned-area **patterns**; TeleViT AUPRC 0.630 vs U-Net++ 0.620 / ViT 0.617 / climatology 0.572 at lead 0; at ~4 months 0.60 vs 0.58. Skill highest in African savannas, **lower in boreal/arid**. Attention: mostly **local** tokens. **Inference:** transformers help slightly at seasonal/global scale; gains are small; not a 1 km early-warning result.
- **Global Probability-of-Fire (PoF)** (ESSOAr preprint): ~9 km daily, weather + vegetation model + ML, skill to 10 days (sometimes 30). Preprint, not the same as Mesogeos.

### 4.6 Spread benchmarks (supporting knowledge only)

- **Huot et al., 2022**, IEEE TGRS, Next Day Wildfire Spread. CONUS 2012–2020, 1 km, next-day **spread** of existing fires. NN > LR/RF on their metric; week-based split. Kaggle + GEE code. **Not occurrence.**
- **Gerard, Zhao, Sullivan, 2023**, NeurIPS, WildfireSpreadTS. 375 m, 5-day inputs, 607 US fires; UTAE AP 0.372; noisy VIIRS labels. **Not occurrence.**
- Follow-ups (e.g. SwinUnet 2025 arXiv:2502.12003) stay on **spread**.

### 4.7 Susceptibility mapping

Large MDPI/ScienceDirect literature: RF, XGBoost, LightGBM, MaxEnt, CNN. **Very high AUC (often >0.90)** is common with **random pixel splits**. **Our inference + multiple spatial-stats papers:** spatial autocorrelation inflates AUC; spatial-block CV is required. California 2025 GeoAI comparison: under **spatial CV**, RF/XGB ~0.83–0.84 ROC, DL ensembles weaker and costlier. These maps answer “where is fire-prone land?” not “will it ignite this week?”

### 4.8 Physics-guided / hybrid

Still early. Reviews (Forests 2025; Zhao) *call for* PINNs and index+ML hybrids. Shmuel 2025 is the cleanest hybrid: **keep FWI structure, retune regionally**, then trees. **Inference:** a useful hypothesis is “GBDT on FWI components + EO features,” not replacing physics wholesale.

---

## 5. Literature comparison matrix

See also the interactive canvas. Columns: year, venue, task, region, resolution, horizon, labels, models, split, headline metric, code.

| Paper | Year | Task | Region | Grid | Horizon | Labels | Models | Split | Headline (source-reported) | Code/data |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jain et al. | 2020 | Review | Global | — | — | — | RF/MaxEnt/ANN dominant | — | 300 papers | Review |
| Huot et al. NDWS | 2022 | **Spread** | CONUS | 1 km | 1 day | MODIS AF next day | Conv AE vs LR/RF | Week 8:1:1 | AUC 0.284 (hard, imbalanced) | Kaggle, GEE |
| Kondylatos et al. GRL | 2022 | **Danger** | Greece | 1 km | 1 day | BA >30 ha | RF, XGB, LSTM, ConvLSTM | Year holdout 2020/21 | F1 ~0.80–0.88; DL > XGB > FWI | FireCube, GitHub |
| Mesogeos | 2023 | Danger + BA given ignition | Mediterranean | 1 km | short-term Track A | EFFIS ignitions/BA | LSTM, Transformer, GTN | Temporal | AUPRC ~0.85–0.87 | GitHub, Drive |
| Gerard et al. WSTS | 2023 | **Spread** | US | 375 m | 1 day | VIIRS AF | U-Net, UTAE, … | LOYO | AP ~0.37 UTAE | NeurIPS |
| Shmuel MLFWI | 2023 | Occurrence vs indices | Multi | weather stations/grid | same-day-ish index | fire occurrence | XGB, RF, MLP vs 14 FWIs | paper protocol | XGB ROC 0.98 / PR 0.94 | paper |
| TeleViT / SeasFire | 2023–26 | Seasonal BA pattern | Global | 0.25° | 0–4 months | FireCCI etc. | ViT, UNet++, TeleViT | Year 2019 test | AUPRC 0.63 vs 0.57 clim. | GitHub |
| Illarionova et al. | 2025 | Occurrence | 4 Russian oblasts | ~650 m | **5 days** | clustered verified hotspots | RF, XGB, ConvLSTM, CNN | region-wise | F1 0.7–0.87; trees often best | paper |
| Shmuel et al. | 2025 | Occurrence vs FWI | Countries | national | weather-index | occurrence | FWI vs regional trees | country | Tree 0.86 vs FWI 0.69 ROC | paper |
| FIRE-RES D1.6 | 2025 | Large-fire occurrence | Europe | EFFIS | early warning | LF / EWE | probabilistic FOP vs FWI | operational eval | Better PR than FWI; precision still low | report |

---

## 6. What is actually state-of-the-art and why?

**Claim of a unique SOTA model is not supported.**

### What *is* supported

1. **For next-day Mediterranean danger on a public datacube:** LSTM / ConvLSTM / Transformer are in a **statistical dead heat**, all beating RF/XGB and FWI **on that protocol** (Kondylatos 2022; Mesogeos 2023). **Best existing *family* for this analog: temporal sequence models on daily 1 km cubes, with tree ensembles as close baselines.**
2. **For multi-region occurrence over 5 days:** **no DL monopoly**; XGBoost/RF are competitive or better and **must be region-adapted** (Illarionova 2025).
3. **For beating operational FWI:** repeatedly shown, but **PR at operational recall is still the bottleneck** (FIRE-RES 2025).
4. **For seasonal global burned-area fields:** TeleViT is a modest, documented improvement over U-Net++/ViT/climatology — **wrong spatial/temporal scale for a local web EWS.**
5. **For spread:** UTAE / later attention U-Nets on WildfireSpreadTS — **wrong task.**

### Comparability problems (why papers claiming SOTA are often incomparable)

- Different **positives** (any hotspot vs >30 ha vs burned area fraction).
- Different **negative sampling** (balanced 2:1 vs full grid). Balanced F1 **does not equal** operational precision.
- **Random spatial CV** in susceptibility vs **temporal year holdout** in FireCube.
- **Same-day weather** vs **forecast weather** (true forecasting).
- Metric shopping: accuracy, ROC-AUC on rare events, F1 on balanced sets.

### External validation

Kondylatos tests an **extreme unseen year (2021)** — stronger than most papers. Mesogeos is the **community benchmark**. Illarionova tests **four climates**. Almost nobody tests **train in Mediterranean, test in boreal** for 1 km daily danger. TeleViT reports **regional skill variation** (savanna vs boreal).

**Our inference:** The “SOTA we should beat” is **not a Transformer**. It is: **(a) FWI, (b) XGBoost on the same features, (c) LSTM-10d on Mesogeos/FireCube-style labels, (d) year and region holdout, (e) PR-AUC + precision at fixed recall + Brier.**

---

## 7. Dataset comparison (practical)

| Source | What | Resolution | Latency / update | History | Access | Role for us |
| --- | --- | --- | --- | --- | --- | --- |
| **Mesogeos** | Harmonized drivers + fires | 1 km, daily | Offline cube 2006–2022 | 17 y | Drive / GitHub | **Best research training cube** for EWS analog |
| **FireCube** | Greece cube | 1 km, daily | 2009–2021 | 13 y | Zenodo | Smaller, published LSTM/ConvLSTM numbers |
| **SeasFire** | Global drivers + BA | 0.25°, 8-day | 2001–2021 | 21 y | Zenodo | Seasonal / transfer experiments, not 1 km EWS |
| **EFFIS** | EU burned area, FWI, Rapid Damage | ~hectares polygons; FWI 8 km | FWI daily; BA delayed | 2000s– | JRC | Labels + operational baseline Europe |
| **NASA FIRMS** | Active fire | 375 m / 1 km | **<3 h** global; minutes US/CA | 2000/2012– | MAP_KEY API | **NRT detection overlay + optional labels** (noisy) |
| **MCD64A1 / FireCCI** | Burned area | 500 m / 250 m | **months** lag | 2000– | NASA/ESA | Gold-ish historical labels; **not NRT** |
| **ERA5 / ERA5-Land** | Weather reanalysis | 0.25° / 9 km | ~5 days (ERA5T faster) | 1950/1940– | CDS, GEE | Training weather; operational needs **forecast** |
| **GFS / ECMWF IFS / open-meteo** | Forecasts | 0.25°–9 km | hours | limited | APIs | **Deployment weather** |
| **MODIS NDVI/LST, SMAP, Copernicus** | Vegetation, LST, soil moisture | 500 m–9 km | 1–8 days | long | GEE, NASA | Fuel/wetness |
| **Sentinel-2** | Optical 10–20 m | 5–10 days, clouds | 2015– | Copernicus | Fine fuel; **not a daily cube** |
| **Landsat** | 30 m | 8–16 days | 1980s– | USGS | History, not NRT |
| **GOES / MSG** | Geostationary fire/wx | 2 km, 5–15 min | NRT | NOAA/EUMETSAT | Detection + nowcast; Americas/Europe/Africa |
| **SRTM / Copernicus DEM** | Elev/slope/aspect | 30 m | static | — | GEE | Static |
| **WorldCover / CORINE / NLCD** | Land cover | 10–30 m | years | — | ESA/EEA/USGS | Fuel type |
| **GPW / WorldPop** | Population | 1 km / 100 m | years | — | SEDAC | Ignition proxy |
| **gridMET / Daymet** | CONUS weather | 4 km / 1 km | 1 day | 1979– | GEE | If US study area |
| **NDWS / WSTS** | ML-ready spread | 1 km / 375 m | offline | 2012–21 | Kaggle/NeurIPS | **Do not train occurrence on these** |

**Best practical combination (justified):**

- **Research/training:** Mesogeos (or FireCube) so we can reproduce LSTM vs XGB vs FWI. Optionally add a **second geography** (e.g. CONUS gridMET + MTBS/MCD64 or Illarionova-style pipeline) for transfer tests.
- **Labels:** EFFIS / MCD64A1 burned area for **large-fire danger**; optionally FIRMS clustered ignitions for **any-occurrence** (noisier). Do not mix without an ablation.
- **Deployment:** ERA5T or NWP forecasts + GEE/MODIS composites + SMAP or modelled soil moisture + static DEM/land cover + FIRMS for concurrent detection. Sentinel-2 as **asynchronous high-res fuel update**, not the daily spine.

---

## 8. Satellite comparison

| Platform | Spatial | Temporal | Strength | Weakness for EWS |
| --- | --- | --- | --- | --- |
| **MODIS** | 250–1000 m | 1–2 / day | Long record, NDVI, LST, fire | Coarse; aging; afternoon gap |
| **VIIRS** | 375 m fire | 2+ / day | Best polar hotspot NRT | Detection, not fuel moisture |
| **Sentinel-2 MSI** | 10–20 m | 5 days | Fuel, NBR, NDVI | Clouds; not daily |
| **Sentinel-3 SLSTR** | 1 km | daily | LST, fire | Coarser optical |
| **Landsat** | 30 m | 8–16 d | Climate-quality history | Too slow for alerts |
| **GOES-R ABI** | 2 km | 5–15 min | Americas nowcast | Not global; detection/wx |
| **Meteosat SEVIRI** | 3 km | 15 min | Europe/Africa | Same |
| **SMAP** | ~9 km | 2–3 d | Soil moisture (Kondylatos: important) | Coarse, freeze/RFI gaps |
| **ERA5 (not a satellite)** | 9–25 km | hourly | Homogeneous wx | Reanalysis delay; not observations |

**Inference:** Daily 1 km danger models are **data-assimilation products** (reanalysis + coarse EO), not raw Sentinel-2 time series. Fine optical helps fuel maps, not the 24 h loop.

---

## 9. Feature comparison

Evidence-backed groups (not an invented wish list):

| Group | Examples | Evidence |
| --- | --- | --- |
| **Weather / VPD** | T, RH, wind, rain, VPD | FWI; Preisler; almost all ML |
| **Dead/live fuel moisture proxies** | FFMC/DMC/DC, NDWI/NDMI, soil moisture | Kondylatos XAI: wetness dominates |
| **Vegetation state** | NDVI, EVI, LAI, land cover | Mesogeos; SeasFire |
| **Drought memory** | 10-day+ precip/T accum, SPEI, PDSI | LSTM 10-day buffer; FWI codes |
| **Topography** | slope, aspect, elevation | susceptibility + spread; weaker for next-day wx-driven ignitions |
| **Human** | population, roads, WUI | Mesogeos; ignition not just weather |
| **Fire weather index** | FWI, ISI, BUI | strong baseline features for GBDT |
| **Historical fire** | frequency, time since fire | susceptibility; **careful leakage** if using recent burns as input to predict same event |
| **Thermal** | LST | useful; **must lag** so it is not the fire itself |
| **Raw RGB Sentinel** | — | little evidence it beats indices for *occurrence* |

Illarionova: **important features differ by region.** Do not freeze one global feature ranking.

---

## 10. Prediction-target analysis (recommended formulation)

### Literature options

1. **Next-day large-fire cell burn** (Kondylatos): operationally relevant (few fires do most damage); ignores small fires.
2. **Next-day any ignition** (FIRMS/EFFIS points).
3. **Fire in next K days** (Illarionova K=5): more usable lead time; easier positives; less “which hour.”
4. **Danger index regression** (predict FWI or burned-area fraction).
5. **Seasonal burned-area map** (SeasFire): planning, not local EWS.

### Recommendation for *this* project

**Primary target:** For each **1 km cell** (or 1–2 km) in the user AOI, predict

> **P(at least one wildfire occurrence affecting the cell in the next H = 3 days)**,  
> trained primarily so that **fires that become ≥30 ha** are weighted more (Mesogeos-style log-area loss),  
> using only information available at issue time **plus NWP forecasts through H**.

**Why 3 days (not 1, not 30):** 1-day matches FireCube but is a weak “early” warning; 5–10 days is used (Illarionova, PoF) but skill and forecast-weather quality decay. Three days is a compromise to be **ablated** against H∈{1,3,5,7}.

**Positives:** cells that intersect a mapped fire (EFFIS/MCD64) with start date in (t, t+H]. Cluster adjacent pixels of the same event (Illarionova/FIRMS clustering) so one fire is not 500 independent positives.

**Negatives:** random cells/days in burnable land cover, **same season**, **spatially buffered** away from positives (e.g. >10 km and not same event). Report metrics both on **sampled** and **full-grid prevalence**.

**Matching:** environmental features at t; fire start date from burned-area product (with known date uncertainty of ±1–several days — Giglio MCD64). **Never** use same-day LST/hotspot as a feature for a same-day label.

**Spatial unit for the product:** model at 1 km; **alert** on aggregated AOI (max or 95th percentile of cell probabilities, plus fraction of cells above threshold) so a user-drawn polygon gets one warning.

---

## 11. Model comparison

| Family | Where it wins (evidence) | Where it fails |
| --- | --- | --- |
| **FWI / NFDRS** | Operational, cheap, understood | No vegetation/human; regional bias |
| **Logistic / GAM** | Interpretable, Preisler-style | Limited interactions |
| **RF / XGBoost / LightGBM / CatBoost** | Tabular occurrence & susceptibility; Illarionova; MLFWI; spatial CV California | Weak native spacetime; need feature engineering for memory |
| **LSTM/GRU** | FireCube next-day; Mesogeos ≈ Transformer | Needs sequences; less spatial context |
| **ConvLSTM** | FireCube precision; some occurrence papers | Cost; Illarionova: not always best |
| **2D CNN / U-Net** | Spread segmentation; seasonal maps | Overkill / leakage if using fire mask |
| **Transformer / ViT / TeleViT** | Mesogeos tie with LSTM; seasonal global small gain | No evidence of large EWS gain; cost |
| **GNN** | Sparse papers; Earth-as-graph SeasFire explorations | Immature for 1 km ops |
| **Ensemble GBDT+LSTM** | Plausible; **not yet standard SOTA** | Must be tested, not assumed |

**Temporal windows:** Kondylatos explicitly used **10 days** (vs Huot 7-day buffer). Illarionova uses sequences for ConvLSTM vs aggregated stats for trees. **We must ablate history ∈ {1, 3, 7, 10, 14} days** rather than assume longer is better (overfitting season).

**Fusion:** FireCube/Mesogeos are **early fusion** (stack variables). Late fusion (weather model × satellite model) is under-compared. **Evidence that multimodal > weather-only:** Kondylatos (RF/XGB already beat FWI before DL). **Evidence that DL fusion > engineered GBDT:** mixed.

---

## 12. Research-gap analysis

Only gaps with **multi-paper support**:

| Existing method | Limitation | Evidence | Why it matters | Proposed improvement | Experiment |
| --- | --- | --- | --- | --- | --- |
| FWI | Weather-only; regional misfit | Shmuel 2025; Kondylatos vs FWI; FIRE-RES PR | Alerts too broad | Add EO+human; regional models | Same labels, FWI vs GBDT vs LSTM |
| Global one-model | Lower skill than regional | Shmuel 2025; Illarionova; TeleViT regional gaps | User AOIs span regimes | Regional heads or fine-tune | Train Med. test other biome |
| Susceptibility RF | Random splits leak space | Jain domain; spatial CV papers; California 2025 drop AUC | Inflated “accuracy” | Blocked spatial + year holdout | Repeat model under both splits |
| Next-day DL papers | Balanced F1 ≠ ops precision | FireCube sampling; FIRE-RES low precision | False alarms | Full-grid PR, FAR vs lead time | Sweep threshold |
| LSTM vs Transformer | Claimed SOTA without clear gap | Mesogeos ~tie | Avoid unjustified complexity | LSTM primary, Transformer ablation | Same cube |
| DL vs GBDT | Opposite winners | Kondylatos vs Illarionova | Architecture cargo-cult | Identical features/splits | Head-to-head |
| Short horizon | “Early” warning is often t+1 | Most DL danger papers | Little prevention time | H=3,5,7 with NWP | Skill vs H |
| Clouds/missing | Optical gaps ignored | Zhao review; FIRMS cloud misses | Ops holes | Masks, persist last clear, weather-only fallback | Ablate missingness |
| Resolution mix | 10 m S2 vs 9 km ERA5 | All cubes resample | False precision | Train at 1 km; don’t pretend 10 m wx | Sensitivity |
| Leakage | Concurrent fire / post-burn NDVI | Huot caveats; detection vs risk mix | Cheating | Feature lag; drop AF from occurrence inputs | Leakage audit |
| Calibration | Softmax ≠ chance of fire | Rare-event PR; few Brier papers | Bad alerts | Temperature scaling / isotonic | Reliability plots |
| XAI | SHAP on trees; IG on LSTM; not user-facing | Kondylatos; Mesogeos future work | Trust | Per-AOI top drivers | Human-readable report |
| Reproducibility | Many papers no cube | vs Mesogeos/FireCube | Can’t beat SOTA fairly | Start on public cube | Mandatory |
| Transfer | Mediterranean-centric DL | FireCube/Mesogeos | Our users may not be Greek | Second continent test | Geographic holdout |

---

## 13. Proposed research hypotheses

**H1 — Multimodal drivers beat FWI** on PR-AUC and precision@recall=0.5 under year holdout. Mechanism: vegetation/wetness/human. Baseline: FWI. Risk: weather already enough in some climates.

**H2 — 7–10 day history beats single-day** for GBDT (engineered lags) and LSTM. Mechanism: fuel drying memory (FWI codes). Baseline: t−1 only. Risk: extra days add noise.

**H3 — LSTM beats XGBoost** on the **same** Mesogeos-style set for next-3-day large-fire labels. Mechanism: nonlinear temporal interactions. Risk: Illarionova-style result (trees win) — then **keep XGBoost as the system model**.

**H4 — Geographic holdout drops skill sharply**; region-specific or adaptation recovers it. Mechanism: different ignition regimes. Baseline: pooled model.

**H5 — Calibrated probabilities + threshold on FAR** improve operational usefulness more than switching LSTM→Transformer. Mechanism: Mesogeos shows architecture ties.

**H6 — Weather-only fallback** when EO is missing retains most of FWI-beating skill. Mechanism: FWI already weather-based; EO is incremental.

**H7 — Transformer/attention fusion** improves H3 by < small margin after H1–H6. Test last.

---

## 14. Proposed model (conditional, experiment-gated)

**Do not implement this as “the architecture” until H1–H3 complete.**

**Stage 0 (mandatory baselines):** FWI; logistic regression; XGBoost/LightGBM on {weather, FWI components, NDVI/SM, DEM, land cover, population, 7–10 day summaries}.

**Stage 1 (strong existing DL analog):** LSTM on 10-day dynamic series + static features (Kondylatos). Optional ConvLSTM if we keep spatial patches.

**Stage 2 (only if Stage 1 wins on PR-AUC *and* geographic holdout is acceptable):** hybrid **stacking**: XGBoost(tabular) + LSTM(sequence) → isotonic calibration. Explain via SHAP on the GBDT and IG/SHAP on LSTM.

**Stage 3 (only if Stage 2 saturates):** Transformer encoder (Mesogeos) or simple feature-attention fusion — **not** ViT/TeleViT unless we also do seasonal products.

This is a **gated pipeline**, not a claim that hybrid is SOTA today.

---

## 15. Baseline strategy

Always, same data/labels/splits/metrics:

1. Climatology (pixel-month fire frequency)
2. FWI (and ISI/BUI)
3. Logistic regression
4. Random Forest
5. XGBoost and LightGBM (CatBoost optional)
6. Then LSTM
7. Then ConvLSTM / Transformer as challengers

No paper is accepted as beaten unless these run.

---

## 16. Data leakage and bias risks

| Leak | How it happens | Prevention |
| --- | --- | --- |
| **Same-location** | Nearby pixels of one fire in train and test | Event-level grouping; spatial buffer; spatial blocks |
| **Temporal** | Random days; 2020 fires in train and adjacent 2020 test | Year holdout; gap days |
| **Post-fire** | NDVI/NBR after burn used as “pre” | Features ≤ t; drop post-fire composites |
| **Concurrent fire** | LST, FRP, hotspot, smoke as features for same-day occurrence | Ban active-fire channels in occurrence model; use only as optional detection layer |
| **Duplicates** | MODIS+VIIRS same event | Cluster events |
| **Future weather** | ERA5 at t+1 treated as observed | Use forecast fields in ops; in research, either forecast or explicitly “perfect prog” ablation |
| **Prevalence bias** | Balanced train → over-alert | Calibrate to real base rate |
| **Prescribed fire** | FIRMS includes agriculture | Land-cover filter; EFFIS forest filter |
| **Suppression** | Labels are “escaped fires” | Acknowledge; don’t claim natural ignition physics |

---

## 17. Class-imbalance strategy

Fires are rare on a 1 km daily grid (often <<1%).

**Preferred:** class-weighted or focal loss; **keep natural prevalence in validation**; tune threshold on PR curve; area-weighted loss (Mesogeos) if targeting large fires.

**Use carefully:** undersampling negatives (FireCube did this — report both sampled and full-grid). SMOTE on spatial pixels is **high-risk** (synthetic geography). Hard-negative mining (high FWI, no fire) is **valuable** (false-alarm reduction).

---

## 18. Evaluation protocol

**Primary:** PR-AUC; **precision at recall ∈ {0.3, 0.5, 0.7}**; false-alarm rate per 1000 km² per day.

**Secondary:** ROC-AUC (not alone); F1 at chosen threshold; Brier; reliability diagrams; lead-time curves (skill vs H).

**Operational question:** *How early can we warn with FAR below an agreed cap?*

**Do not optimize accuracy.**

---

## 19. Generalisation strategy (final benchmark)

**Required folds:**

1. **Temporal:** last 1–2 fire seasons held out (FireCube pattern).
2. **Leave-one-year-out** if compute allows.
3. **Geographic:** train on subset of Mesogeos countries / ecoregions, test unseen ecoregion.
4. **Seasonal:** train spring–summer, test extreme late season if samples allow.

**Final claimed benchmark (recommendation):**

- Train: Mesogeos years T0–T1 except held-out countries C*
- Val: year T2
- Test A: year T3, **seen** countries (temporal)
- Test B: all years held-out **unseen** country/ecoregion (spatial)
- Optional Test C: non-Mediterranean cube

A model that only wins Test A is **not** deployable worldwide.

---

## 20. Uncertainty strategy

- Isotonic or Platt calibration on validation years.
- **Ensemble** of GBDT seeds + optional LSTM dropout at test (approximate epistemic).
- **Aleatoric:** Kondylatos notes overlapping burned/unburned distributions — ignition is stochastic; output **probability**, not binary only.
- Optional: conformal risk control for FAR caps.

---

## 21. Explainability strategy

User question: *Why is this area high-risk now?*

- **GBDT:** SHAP — top 5 drivers with values vs regional climatology (“RH 18% vs typical 45%; 7-day rain 0 mm; soil moisture percentile 5”).
- **LSTM:** Integrated Gradients / temporal contribution (Kondylatos-style: which days).
- **Do not** lead with Grad-CAM on Sentinel RGB unless the model actually uses imagery.
- Map: probability + **driver-change** small multiples.

---

## 22. Real-time feasibility

| Component | Feasible? | Notes |
| --- | --- | --- |
| Daily 1 km GBDT inference | Yes | CPU, seconds per AOI |
| LSTM 10-day | Yes | Small nets; GPU optional |
| ConvLSTM full country | Heavy | Restrict to AOI patches |
| TeleViT global 0.25° | Research | Wrong scale |
| ERA5T + GFS | Yes | Hours of latency |
| Sentinel-2 daily | No | 5-day + clouds |
| FIRMS overlay | Yes | <3 h |
| MCD64 labels | Training only | months lag |
| Missing EO | Need fallback | Weather-only model |

**Cost:** GEE + Open-Meteo/NWP + FIRMS is the realistic student/research ops stack. ECMWF commercial high-res is optional.

---

## 23. Proposed system architecture (conceptual)

```
User AOI
  → scheduler (e.g. 6–24 h)
  → NWP weather + GEE composites (NDVI/SM/LST lagged) + static layers
  → quality flags (cloud, stale)
  → baseline FWI + XGBoost + (if justified) LSTM
  → calibrated P_fire(H=3d) per cell
  → AOI aggregate + threshold
  → map + SHAP text + alert
  → FIRMS layer (detection, secondary)
  → log outcomes for later labels
```

Weather-only path if GEE fails.

---

## 24. Experiment plan (order)

1. Reproduce Mesogeos Track A LSTM vs XGBoost vs FWI (sanity).
2. Relabel to H=3 large-fire; compare.
3. History ablation 1/3/7/10/14.
4. Full-grid PR vs balanced F1.
5. Missing-data / weather-only.
6. Geographic holdout.
7. Calibration + FAR–lead-time plot.
8. SHAP qualitative vs known fire weather.
9. Only then Transformer / ConvLSTM / stacking.

Stop and **ship GBDT** if H3 fails.

---

## 25. Implementation roadmap (after this report)

1. Freeze target/horizon/grid (Section 10).  
2. Build or download Mesogeos; training scripts.  
3. Baselines (Section 15).  
4. Experiments (Section 24).  
5. If improvement holds: AOI web dashboard, alerts, FIRMS overlay.  
6. Second geography before claiming generality.

No “final neural architecture” sprint before step 4.

---

## 26. Expected limitations

- Cannot forecast a single discarded cigarette; model **conditions**, not all ignitions.
- Burned-area dates are uncertain; small fires missing.
- Suppression and policy change labels.
- Cloud-limited optical.
- Mediterranean-trained models may fail in boreal peat or tropical deforestation fire.
- High recall ⇒ many false alarms (FIRE-RES).
- Perfect-prog weather in research overstates ops skill.

---

## 27. References (priority)

1. Jain, P. et al. (2020). A review of machine learning applications in wildfire science and management. *Environmental Reviews*. https://doi.org/10.1139/er-2020-0019  
2. Kondylatos, S. et al. (2022). Wildfire danger prediction and understanding with deep learning. *GRL* 49. https://doi.org/10.1029/2022GL099368  
3. Kondylatos, S., Prapas, I., Camps-Valls, G., Papoutsis, I. (2023). Mesogeos. NeurIPS Datasets. https://arxiv.org/abs/2306.05144 — https://github.com/Orion-AI-Lab/mesogeos  
4. Prapas, I. et al. FireCube. Zenodo. https://doi.org/10.5281/zenodo.6475592  
5. Huot, F. et al. (2022). Next Day Wildfire Spread. *IEEE TGRS*. https://arxiv.org/abs/2112.02447  
6. Gerard, S., Zhao, Y., Sullivan, J. (2023). WildfireSpreadTS. NeurIPS.  
7. Alonso, L. et al. (2025). SeasFire cube. *Scientific Data*. https://doi.org/10.1038/s41597-025-04546-3  
8. Prapas, I. et al. TeleViT. ICCVW 2023; TeleViT1.0 2026. https://arxiv.org/html/2512.00089  
9. Illarionova, S. et al. (2025). Robust wildfire occurrence prediction. *Sci. Rep.* https://doi.org/10.1038/s41598-025-94002-4  
10. Shmuel, A. et al. (2023). ML-based fire weather indices. https://iopscience.iop.org/article/10.1088/2632-2153/acc008  
11. Shmuel, A. et al. (2025). Fire weather indices tailored to regional patterns. https://doi.org/10.1038/s44304-025-00126-y  
12. Zhao, Y. et al. Deep Learning for Wildfire Risk Prediction. https://arxiv.org/abs/2405.01607  
13. Van Wagner, C. E. (1974). Structure of the Canadian Forest Fire Weather Index.  
14. Pettinari, M. L. & Chuvieco, E. (2020). Fire danger observed from space.  
15. Preisler, H. K. et al. (2004). Probability-based models for estimation of wildfire risk.  
16. Giglio, L. et al. MODIS burned area (MCD64).  
17. NASA FIRMS. https://firms.modaps.eosdis.nasa.gov/  
18. EFFIS. https://effis.jrc.ec.europa.eu/  
19. FIRE-RES (2025). D1.6 Early-warning indicators. https://fire-res.eu/  
20. Ecological Informatics (2025). ML pipeline survey. https://doi.org/10.1016/j.ecoinf.2025.103325  
21. Forests (2025). AI forest fire prediction review. https://www.mdpi.com/1999-4907/16/4/704  
22. Ghali, R. & Akhloufi, M. (2023). DL wildland fires satellite. *Fire* 6(5):192.  

---

*End of research phase document. Next phase: freeze the prediction target and run baseline experiments on a public cube — still not “the final model.”*
