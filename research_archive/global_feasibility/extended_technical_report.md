# Extended Greece–BC technical audit — 11 October 2026

**Overall gate: FAIL — label/calendar compatibility remains unresolved.** Weather inventories and recent numeric cutouts support the provisional UTC contract; actual historical availability at the cutoff remains unproven. The static sample now comprises three contrasting cells per region. Completed earlier samples were reused, not recomputed.

## Forecast contract and evidence

Provisional issue: **12:00 UTC**; input cutoff: **11:00 UTC**; target: **following UTC calendar day [00:00,24:00)**. Candidate cycle: same-day **06Z**, conditional on verified delivery. Initialization is not publication. A same-day 12Z cycle cannot satisfy an 11Z cutoff.

For a 06Z run, instantaneous TMP/RH/U/V samples at **f018–f041** span the target day's 00:00–23:00 hours. Precipitation intervals end at **f019–f042**; f042 is the next midnight endpoint, excluded from instantaneous daily sampling but required to finish the precipitation day.

| Run / season | Target UTC day | Hourly inventories f018–f042 | Latest parent-GRIB S3 Last-Modified |
|---|---|---:|---|
| 2025-01-15 06Z / winter | 2025-01-16 | 25/25 | 09:45:14 UTC on run day |
| 2025-07-15 06Z / summer | 2025-07-16 | 25/25 | 09:58:35 UTC on run day |
| 2026-04-15 06Z / spring | 2026-04-16 | 25/25 | 09:55:24 UTC on run day |
| 2026-10-09 06Z / autumn | 2026-10-10 | 25/25 | 09:55:11 UTC on run day |

**Verified archive evidence:** all 100 index objects and parent-GRIB HEAD requests succeeded; zero missing required field types. All initialization identifiers, instantaneous leads and required accumulation descriptors matched. Four 512-byte partial GRIB reads verified the temperature grid is 1440×721, 0.25°, covering both regions (90°N–90°S, 0–359.75° longitude). BC longitudes require the equivalent 0–360 representation when indexing this grid.

**Timing distinction:** all sampled S3 object modification timestamps precede 11Z. These are archive-object metadata, not independently observed successful public retrievals at that time. Neither index timestamps, initialization times nor present-day HTTP success prove the historical cutoff. No cutoff pass is assigned. [S3 HEAD semantics](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html); [NOAA archive and new-object notification source](https://registry.opendata.aws/noaa-gfs-bdp-pds/).

**Numeric verification:** for the October run, retrieved 25 tiny regional files per region, each a 3×3-point box. All 24 target-day TMP/RH/U/V time slots and 24 precipitation intervals were decoded with no absent field slots or missing-value bitmap. An independent GDAL decode of **350 bands** confirmed all valid timestamps and numerical agreement within **0.000021 native units**. Earlier seasons/years have verified inventory and geographic support only: their regional values and numeric missingness were not downloaded or validated. This is not a four-season numerical completeness claim.

## Units and precipitation

Temperature: **K** (°C = K−273.15); 2 m relative humidity: **%**; 10 m U/V wind: **m/s** (speed = sqrt(U²+V²)); total precipitation: **kg/m²**, numerically equivalent to **mm** water. [NCEP moisture parameter definitions](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-1.shtml).

GRIB template 4.8 explicitly identifies accumulated precipitation and its ending timestamp. These files contain both running six-hour-block accumulations and accumulations since initialization. Derive hourly rain by differencing within the same block; at a block reset use the new one-hour accumulation directly. Do not sum overlapping cumulative products. The saved verifier checks all intervals and does not silently clip negative increments.

For 2026-10-10, gridpoint daily rain totals were **2.1875–10.4375 mm in Greece** and **0.5625–2.1875 mm in BC**, with no negative hourly increments. Summed block increments differ from the alternative full-run endpoint difference by at most **0.0625 mm / 0.125 mm**, respectively. This is consistent with separate product packing/rounding; independent decoding reproduces the values. Preserve the discrepancy and use one documented accumulation method consistently.

## Expanded static sample

Reused the original two cells and added mountain/forest and coastal cells in each region. All use the existing EPSG:6933 grid (25,000 m spacing, origin 0,0), native-pixel centre selection and the same ellipsoidal row-area weighting, source products and class mapping. No class interpolation. Source NoData remains separate from water.

| New cell | Mean DSM elevation (m) | Tree cover | Water | DEM / land-cover valid area |
|---|---:|---:|---:|---:|
| Greece mountain `82_187` | 958.4 | 84.86% | 0.01% | 100% / 100% |
| Greece coastal `91_181` | 238.1 | 42.93% | 21.37% | 100% / 100% |
| BC mountain `-473_226` | 1451.1 | 67.19% | 4.98% | 100% / 100% |
| BC coastal `-478_222` | 98.9 | 24.01% | 73.27% | 100% / 100% |

Full cell IDs carry the `EPSG6933_25km_` prefix. Geometry, all 11 land-cover fractions, NoData, native pixel spacing, tile URLs, ETags and modification times are in the new static manifest. The sources remain Copernicus GLO-30 Public DSM and ESA WorldCover **2021 v200**. This purposive six-cell sample establishes local technical feasibility, not representative accuracy or province/country-wide completeness. Elevation means include valid water pixels; land-only summaries would require an explicit common mask. GLO-30 is a surface model and the AWS instance does not expose a pinned scientific release in these URLs; ETags identify the sampled objects. WorldCover 2021 is dated land cover, not current fuel state.

## Saved outputs and remaining limits

- `extended_weather_manifest.json`: selected index records, SHA-256s, parent-object ETags/timestamps, valid times, grid-header evidence, units, all decoded cutouts and errors.
- `audit_weather_cutouts/`: 50 tiny original GRIB2 responses, retained for offline verification. Weather response bodies total approximately **4.20 MB**, mostly text indices; no full forecast grids were downloaded.
- `extended_static_manifest.json`: four new native-pixel window summaries; earlier two remain in `static_sample_manifest.json`.
- `extended_audit_summary.json`: validation counts and numerical checks.
- `extend_technical_audit.py`: bounded acquisition/processing; `summarize_extended_audit.py`: offline validation. Dependencies are Python 3.12, numpy 2.3.5, rasterio 1.5.2 (GDAL 3.12.2), pyproj 3.8.0. The earlier static processing script is imported without modifying it. NOMADS retention limits future re-downloads; retained cutouts allow offline rechecking.

Outstanding: historical first-public-availability evidence; numeric regional coverage outside the October run; exact operational GFS model-build provenance by year; broader static coverage and source age. No performance, calibration or operational reliability claim follows from this sample.

**Single next action:** capture a small prospective 06Z availability trace for f018–f042 through the 11Z cutoff, logging first successful retrieval, field completeness, URL and content hash. This will test the delivery contract while agency label/calendar clarification proceeds independently.
