# Two-cell common-grid static-data check

**Result: technically feasible for these two sample cells only.** This does **not** change the overall Greece–BC feasibility gate, which remains **FAIL pending label/calendar clarification**.

## Grid and sources

- One 25,000 m × 25,000 m cell per region (625 km²), EPSG:6933, origin (0, 0) m; ID is the floor-divided projected X/Y index. Exact bounds and WGS84 polygons are in [`static_sample_manifest.json`](static_sample_manifest.json).
- Elevation: [Copernicus DEM GLO-30 Public](https://copernicus-dem-30m.s3.amazonaws.com/readme.html), the same public COG product in both locations, nominal 1 arcsecond (~30 m). It is a **digital surface model**, including canopy/buildings, not bare-earth elevation. The two 1° tiles were accessed by HTTP byte-range; object ETags, Last-Modified values and URLs are in the manifest.
- Land cover: [ESA WorldCover 2021 v200](https://esa-worldcover.org/en/data-access), same 10 m nominal COG and official 11-class code mapping in both locations. This is a fixed 2021 snapshot, not a current-year fuel update. The two 3° tiles were read by byte-range; object metadata is in the manifest.
- Processing: source-pixel centres inside the EPSG:6933 cell, **no interpolation or class resampling**; WGS84 ellipsoidal area per source latitude row weights elevation mean/std and class fractions. Valid-area fractions exclude source NoData; NoData coverage is reported separately. Native-pixel centre inclusion approximates the boundary by at most a pixel. Same script and versions for both cells: [`compute_static_sample.py`](compute_static_sample.py).

| Sample cell | Elevation mean (m) | Elevation range (m) | Selected land-cover fractions of valid area | Valid coverage |
|---|---:|---:|---|---:|
| Greece `EPSG6933_25km_86_186` | 132.5 | 52.5–482.8 | cropland 79.03%; grassland 9.22%; built-up 6.33%; tree cover 4.12% | DEM 100%; WorldCover 100% |
| BC `EPSG6933_25km_-466_226` | 1173.5 | 336.0–1893.6 | tree cover 58.16%; grassland 34.57%; permanent water 5.12% | DEM 100%; WorldCover 100% |

All 11 source class fractions, unmapped-code fraction, polygon geometry, native resolution, window size, NoData counts/fractions, tile URLs and source-object metadata are recorded in the machine-readable manifest. Fractions include non-burnable cover by design; a later fire-fuel transformation would be a separate, explicitly defined feature.

## Reproduction and limitations

Run the script with Python 3.12 plus `numpy`, `rasterio` and `pyproj`; the tested library versions are recorded in the manifest. It reads only the four intersecting COG windows, not full tiles. The public remote objects must remain accessible; ETags detect source replacement but the sample does **not** archive raster bytes. No global coverage assessment was performed. GLO-30 Public has known tile-release gaps in some countries; those do not affect the two sampled cells. A 2021 land-cover snapshot can become stale after land-use change or fire, and cannot be used as a temporally honest pre-2021 feature. DEM vertical datum and DSM-vs-terrain suitability would need review before interpreting elevation as topographic height.

**Next static-data step if the label gate later passes:** test additional cells in each region, including coast/water, mountains and product-edge locations; freeze tile snapshots/checksums and decide how to handle stale land cover for the forecast issue year.
