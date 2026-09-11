# Dataset Inspection Report

**Date of inspection:** 2026-09-11  
**Raw file:** `data/raw/Algerian_forest_fires_dataset_UPDATE.csv`  
**Source:** UCI Machine Learning Repository Dataset 547 (official zip)  
**SHA-256:** `3be9ea1575c093cf6bfa7b56110cedc827aa8ca7953bf571530f8cd9990e0a0c`  

No model was trained. No target labels were remapped beyond whitespace inspection. The raw CSV was not modified.

## 1. Filename and file structure

- **Filename:** `Algerian_forest_fires_dataset_UPDATE.csv`
- **Size:** 14,759 bytes
- **Raw line count:** 249 lines (includes titles, headers, blank line, CRLF endings)

Structural markers in the raw file:

| Line (0-based) | Content |
|----------------|---------|
| 0 | `Bejaia Region Dataset` (region title, not data) |
| 1 | Column header |
| 2–123 | Bejaia region data rows (122 rows) |
| 124 | Blank line |
| 125 | `Sidi-Bel Abbes Region Dataset` (region title) |
| 126 | Repeated column header |
| 127–248 | Sidi-Bel Abbes data rows (122 rows) |

Naive `pd.read_csv(path)` fails structurally (treats the region title as the only column). Loading with `header=1` yields a usable frame plus 2 non-data rows from the mid-file markers.

## 2. Shape

| View | Shape |
|------|-------|
| After `header=1` (includes 2 structural junk rows) | 246 × 14 |
| Data-like rows only (`day` is numeric) | **244 × 14** |
| Valid fully numeric rows after excluding 1 malformed row | **243 × 14** (+ derived `Region`) |
| UCI metadata claim | 244 instances |

## 3. Columns (after stripping header whitespace)

Raw header contains leading/trailing spaces on some names (` RH`, ` Ws`, `Rain `, `Classes  `).

| Column | Role |
|--------|------|
| `day` | Date part |
| `month` | Date part |
| `year` | Date part |
| `Temperature` | Weather feature (°C, noon) |
| `RH` | Relative humidity (%) |
| `Ws` | Wind speed (km/h) |
| `Rain` | Rain (mm) |
| `FFMC` | Fine Fuel Moisture Code (FWI) |
| `DMC` | Duff Moisture Code (FWI) |
| `DC` | Drought Code (FWI) |
| `ISI` | Initial Spread Index (FWI) |
| `BUI` | Buildup Index (FWI) |
| `FWI` | Fire Weather Index |
| `Classes` | **Target** (`fire` / `not fire`) |

`Region` is **not** a CSV column; it must be derived from the two file sections (Bejaia / Sidi-Bel Abbes).

## 4. Data types (as loaded)

All columns initially load as **object/string** because of junk rows and whitespace. After excluding structural/malformed rows and coercing:

| Column | Observed dtype |
|--------|----------------|
| `day`, `month`, `year`, `Temperature`, `RH`, `Ws` | integer |
| `Rain`, `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI` | float |
| `Classes` | categorical string |
| `Region` (derived) | categorical string |

`year` is constant **2012** for all valid rows.

## 5. Missing values

In the `header=1` frame:

- Mid-file region title row: all feature/target fields missing
- Repeated header row: not missing, but non-numeric labels
- **One data row (index 167 in that frame)** has `Classes = NaN` because fields are shifted (see malformed values)

UCI page text says “Has Missing Values? No”, but the **packaged CSV** has formatting defects that create apparent missing/shifted values.

## 6. Duplicate rows

After stripping string whitespace on data-like rows: **0 exact duplicate rows**.

## 7. Invalid / malformed values

### 7.1 Structural non-data rows

- Region title row (`Sidi-Bel Abbes Region Dataset`)
- Repeated header row (`day,month,year,...`)

### 7.2 Malformed observation (raw file line with `14.6 9`)

Raw fields observed:

`14,07,2012,37,37,18,0.2,88.9,12.9,14.6 9,12.5,10.4,fire`

Interpretation:

- `DC` contains `"14.6 9"` (space inside field)
- `FWI` contains `"fire"`
- `Classes` is missing

**Likely intended shift (hypothesis, not applied):**  
`DC=14.6`, `ISI=9`, `BUI=12.5`, `FWI=10.4`, `Classes=fire`

This would restore the UCI-reported class counts (138 fire / 106 not fire). **Not applied yet — needs approval.**

### 7.3 Target whitespace variants (semantics unchanged)

Observed raw `Classes` spellings on data rows include whitespace variants such as `"fire"`, `"fire "`, `"not fire"`, `"not fire   "`, etc. After `.strip()` only, labels are still **`fire`** and **`not fire`** (no renaming of class meaning).

## 8. Target distribution

| Classes (whitespace-stripped) | Count | Notes |
|-------------------------------|-------|-------|
| `fire` | 137 | excluding malformed row |
| `not fire` | 106 | |
| unreadable (`NaN` on malformed row) | 1 | |
| **UCI claimed** | **138 fire / 106 not fire** | matches if malformed row is repaired as `fire` |

By region (243 valid rows):

| Region | fire | not fire |
|--------|------|----------|
| Bejaia | 59 | 63 |
| Sidi-Bel Abbes | 78 | 43 |

Mild imbalance overall; stronger fire rate in Sidi-Bel Abbes.

## 9. Categorical variables

| Variable | Values | Notes |
|----------|--------|-------|
| `Classes` (target) | `fire`, `not fire` | strip whitespace only |
| `Region` (derived) | Bejaia, Sidi-Bel Abbes | from file sections |
| `month` | 6, 7, 8, 9 | numeric codes; June–September 2012 |
| `day` | 1–31 | calendar day |
| `year` | 2012 only | constant; not useful as a feature |

## 10. Numerical variables

Weather: `Temperature`, `RH`, `Ws`, `Rain`  
FWI system: `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`

Observed ranges on 243 valid rows (approx.):

| Feature | Min | Max |
|---------|-----|-----|
| Temperature | 22 | 42 |
| RH | 21 | 90 |
| Ws | 6 | 29 |
| Rain | 0.0 | 16.8 |
| FFMC | 28.6 | 96.0 |
| DMC | 0.7 | 65.9 |
| DC | 7.0 | 220.4 |
| ISI | 0.0 | 19.0 |
| BUI | 1.1 | 68.0 |
| FWI | 0.0 | 31.1 |

Note: UCI text lists FFMC max 92.5; observed max is **96.0**. Keep as measured unless documentation says otherwise.

## 11. Potential data-leakage / fairness risks

1. **FWI composite redundancy (important):** `FWI` is an index computed from other FWI components. Using `FWI` together with `FFMC/DMC/DC/ISI/BUI` is not label leakage, but it is highly redundant and can make classification look unrealistically easy.
2. **Strong feature–label association:** On valid rows, `FFMC`, `ISI`, and `FWI` show high correlation with the fire label. This is expected for fire-danger indices, but the viva should state that the model predicts from weather/FWI conditions, not from satellite images or live sensors.
3. **`day` / exact calendar identity:** Using raw `day` may memorize calendar artifacts in a tiny 2012-only sample. Prefer `month` (seasonality) over `day`, or drop both date parts after discussion.
4. **No train/test leakage yet:** no model trained; future split must be stratified and done **after** cleaning, with the same preprocessing pipeline used at prediction time.
5. **Region imbalance:** class rates differ by region; stratify by label (and optionally check region performance).

## 12. Highly correlated / redundant features

Pearson correlations on 243 valid rows (|r| ≥ 0.90):

| Pair | Correlation |
|------|-------------|
| DMC vs BUI | 0.982 |
| DC vs BUI | 0.942 |
| ISI vs FWI | 0.923 |

Also high (|r| ≥ 0.85): DMC–DC, DMC–FWI, BUI–FWI.

This matches FWI system definitions (BUI derived from DMC/DC; FWI from ISI/BUI, etc.).

## 13. Cleaning required (before any training)

Required to make the file analyzable:

1. Skip region title lines and the repeated mid-file header; keep both regions’ data rows.
2. Strip whitespace from column names and from `Classes` values (**no class renaming**).
3. Derive `Region` from file section.
4. Coerce numeric columns to numeric types.
5. Decide how to handle the **one malformed row** (repair with documented hypothesis **or** drop).
6. Drop constant `year`.
7. Decide feature set regarding FWI redundancy / possible drop of `FWI` and/or `BUI`.
8. Encode `Classes` for modeling as 0/1 **only after** cleaning, keeping mapping explicit: `not fire → 0`, `fire → 1` (or equivalent), without changing meanings.
9. Save cleaned data to `data/processed/`; never overwrite `data/raw/`.

## 14. Proposed preprocessing decisions (awaiting approval)

See the proposal in the project status message / PR description. No preprocessing file has been written yet beyond this report.
