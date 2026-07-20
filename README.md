# GeoCoreFusion

> **V6.1 visual-full-detail release (2026-07-20).** V6.1 returns to the direct V6 logic: after the latest registration front end and edge-preserving RGB denoising, all reconstructed RGB spatial detail is allowed to enter the NIR/SWIR estimate. The `scientific_conditional`/V8 gate is not part of this product path.

**Version:** V6.1 visual-full-detail benchmark, package `0.6.1`

**Release tag:** `v0.6.1-v61`

GeoCoreFusion is a research prototype for close-range drill-core RGB-NIR-SWIR registration, spectral harmonization, and spatial fusion. V6.1 targets the practical goal requested for this project: recover substantially more fractures, grains, bedding, dark-region texture, tray edges, labels, and other registered RGB spatial detail while cycling the final fused product back toward the measured low-resolution NIR/SWIR observations.

The working paper title is:

> GeoCoreFusion V6.1: Subpixel-Capable Registration and Denoised Visual-Full-Detail Fusion for RGB-NIR-SWIR Drill-Core Imagery.

## Scope And Claim Boundaries

The continuous product is defined on the RGB spatial grid over **691-2518 nm**, sampled at about **5 nm** for **367 bands**. V6.1 deliberately transfers all denoised, registered RGB spatial detail, including RGB-only texture, shadows, reflections, coloured marks, and labels. The resulting high-resolution pixels are therefore **RGB-textured NIR/SWIR estimates**, not independent high-resolution spectrometer measurements.

| Question | Current evidence | Defensible conclusion |
|---|---|---|
| Can the registration code estimate transforms below one analysis pixel? | Deterministic known-truth TRE/EPE | Yes, at algorithm-component level on the synthetic analysis grid |
| Is real RGB-NIR-SWIR alignment already below 0.5 native RGB pixel? | Same-data structure scores and overlays | Not yet proved; an independent target/landmark TRE audit is still required |
| Does V6.1 transfer more useful RGB spatial detail than V6? | Shared-reflectance-domain full/crop comparisons, reliable-detail correlation, edge F1, and boundary correlation | Yes, on both tested real ROIs |
| Are dark textured regions improved without simply amplifying flat noise? | Dark reliable-detail correlation and RGB-flat high-frequency energy | Yes on both tested ROIs; the flat-area energy diagnostic is lower than V6 |
| Are the original spectral curves retained? | The actual final HR product is PSF-degraded back to the original NIR/SWIR grid | Yes at the measured low-resolution observation scale; this is not independent HR-SWIR truth |
| Does the current evidence prove true high-resolution SWIR texture? | No independent HR-NIR/SWIR reference | No |

Independent high-resolution NIR/SWIR ROIs, calibrated targets, repeated scans, point spectroscopy, XRD/Raman, and expert mineral labels remain necessary for paper-level physical validation.

## V6.1 Method

V6.1 retains the corrected V6 registration front end and replaces weak shared-detail extraction with a direct, denoised, multiscale visual-full-detail route.

### Registration

1. Global structure-map alignment initializes RGB, NIR, and SWIR geometry.
2. ROI affine refinement, column geometry, and guarded local tie points refine the alignment on the low-resolution observation grid.
3. Raster endpoint scaling now maps pixel centers with `(target_size - 1) / (source_size - 1)` instead of a size ratio, eliminating systematic endpoint drift.
4. Affine and remap scoring use explicit finite-data support masks. Invalid borders remain invalid rather than being synthesized by reflected padding.
5. Tie-point peaks on the search boundary are rejected because they cannot support an interior subpixel estimate.
6. Known-truth tests report target-registration error (TRE) and dense end-point error (EPE), separately from same-data structure correlation.

### Fusion

1. NIR/SWIR overlap harmonization and the low-rank material field remain observation-derived.
2. RGB is edge-preserving denoised before detail extraction (`visual_detail_denoise_strength: 0.55`).
3. Four spatial scales (`0.65`, `1.35`, `2.80`, `5.60` px) capture fine grains, fractures, bedding, tray edges, and larger local contrast.
4. Luminance and chroma gradient candidates are combined by strongest-gradient selection, so equal-luminance colour edges are no longer lost.
5. A screened-Poisson solve reconstructs an integrable detail field from the selected gradients instead of stacking raw high-pass residuals.
6. Dark-region relative contrast is boosted in the log domain; a bounded log gain (`0.92`) and limited additive detail (`0.22`) inject the reconstructed field.
7. Gain limits (`0.52-1.92`), reflection padding, and controlled sharpening (`0.25`) reduce clipping, ringing, and boundary artefacts.
8. Four final-product back-projection iterations cycle the actual modulated HR product toward the measured low-resolution NIR/SWIR cube.
9. The output contract is the single `geocorefusion.visual-full-detail.v1` product. The V8 conditional gate is not called.

## Registration Benchmark

The seeded benchmark uses known synthetic transformations to test algorithm components. It is not a replay of the production ROI configuration, and bitwise identity is not promised across OpenCV builds. Errors are in **synthetic analysis-grid pixels**, not native RGB pixels.

| Stage | Median error | P95 error | Additional evidence |
|---|---:|---:|---|
| Coarse affine TRE | 0.1494 px | 0.2307 px | Maximum 0.2700 px |
| ROI affine TRE | 0.2025 px | 0.2832 px | Maximum 0.3075 px |
| Dense residual EPE | 0.1850 px | 0.4289 px | 104 tie points; maximum 1.0589 px |

Dense structure correlation increased from **0.7379 to 0.9266**. These results validate subpixel behavior on known synthetic truth only. They do not establish a real-data claim of `<0.5` RGB pixel: the current real alignment is optimized and scored on an approximately 160 x 293 HSI analysis grid, where one analysis pixel corresponds to several RGB pixels.

## V6 → V6.1 Real-ROI Results

V6 and V6.1 display bands were reconstructed from saved coefficients in reflectance space and mapped with one shared per-scene linear display domain. This avoids comparing independently stretched PNGs.

| Dataset | Variant | RGB/material boundary | 2201 nm reliable ρ | 2201 nm dark reliable ρ | 2201 nm edge F1 | Final-product RMSE | SAM (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| 3DSSZ | V6 | 0.332 | 0.400 | 0.359 | 0.547 | 0.00462 | 0.923 |
| 3DSSZ | **V6.1** | **0.459** | **0.734** | **0.820** | **0.626** | **0.00386** | **0.872** |
| ZKH3 | V6 | 0.246 | 0.489 | 0.416 | 0.643 | 0.01453 | 0.917 |
| ZKH3 | **V6.1** | **0.324** | **0.636** | **0.582** | **0.733** | **0.00946** | **0.682** |

The 2201 nm RGB-flat high-frequency energy ratio fell from **11.13 to 1.57** on 3DSSZ and from **2.39 to 1.55** on ZKH3. Together with the crops, this indicates that the new route replaces much of V6's non-RGB-aligned speckle with more coherent registered structure instead of merely increasing sharpening strength.

Natural-scene edge-width/overshoot values remain diagnostic only. Visual inspection found no obvious destructive double edges or ringing in the selected full/crop views, but a publishable halo/MTF claim still requires a calibrated slanted-edge target.

The machine-readable summary is `artifacts/v61_research/experiments/benchmark_summary.json`; publication figures are under `artifacts/v61_research/figures/`.

## Repository Contents

```text
configs/                   V3-V6.1 ROI configurations and benchmark exports
docs/                      Method, registration, validation, and review notes
scripts/                   Benchmarks, diagnostics, figures, and output validators
src/geocorefusion/         Python package implementation
tests/                     Unit and deterministic synthetic tests
artifacts/v61_research/    V6.1 evidence, shared-domain figures, and final report assets
```

Large local experiment products remain outside normal Git history:

- `data/`: raw RGB/NIR/SWIR observations
- `runs/`: intermediate and ROI run outputs
- `result/`: full ENVI/BigTIFF exports, about 2.2-2.3 GiB per cube
- `figures/`: ad hoc figure exports outside the versioned V6.1 artifact package

Use a release asset, research-data repository, object storage, or a project archive for full data products.

## Install

```powershell
conda env create -f environment.yml
conda activate geocorefusion
python -m pip install -e .
```

If the active environment already has the required scientific packages:

```powershell
python -m pip install -e . --no-deps
```

## Run V6.1

Inspect source data:

```powershell
geocorefusion inspect "E:\Code\GeoCoreFusion\data\2023_09_04_14_34_41-3DSSZ-1-16-267.0_276.0-532.0_552.0"
```

Run the V6.1 ROI workflows:

```powershell
python -m geocorefusion.cli run configs\3dssz_roi_fusion_v61.yaml
python -m geocorefusion.cli run configs\zkh3_roi_fusion_v61.yaml
```

The release configurations write metrics and previews but not the approximately 2.3 GiB full cube. Set `output.write_envi: true` only for final export. The `*_v61_benchmark.yaml` configurations additionally retain material coefficients so the publication figures can reconstruct V6/V6.1 in one shared reflectance domain.

Rebuild the shared-domain figures and benchmark summary after running the two benchmark configurations:

```powershell
python scripts\build_v61_figures.py --output-dir artifacts\v61_research\figures --summary artifacts\v61_research\experiments\benchmark_summary.json
```

Validate a complete ENVI output directory:

```powershell
geocorefusion validate "E:\Code\GeoCoreFusion\runs\<run-name>"
```

## Output Layout

Depending on the output flags, a run contains:

```text
runs/<run>/
├─ cube/fused_continuous_691_2518nm.hdr|dat
├─ coefficients/material_coefficients.hdr|dat
├─ analysis/harmonized_lowres.hdr|dat
├─ metrics/quality_report.json
├─ metrics/spatial_uncertainty.hdr|dat
├─ metrics/spatial_detail_gain.hdr|dat
├─ metrics/spatial_additive_detail.hdr|dat
├─ metadata/
│  ├─ registration_model.json
│  ├─ spectral_harmonization.json
│  ├─ psf_model.json
│  ├─ subspace_model.json
│  ├─ fusion_model.json
│  ├─ band_metadata.csv
│  └─ processing_config.json
├─ previews/
└─ manifest.json
```

New runtime manifests generated by package `0.6.1` record that version and the `geocorefusion.visual-full-detail.v1` single-product contract. Existing run directories retain the software-version provenance with which they were generated.

## Implementation Map

- Registration, validity masks, endpoint geometry, and guarded tie points: `src/geocorefusion/registration.py`
- NIR/SWIR spectral harmonization: `src/geocorefusion/spectral.py`
- RGB denoising, multiscale gradients, screened-Poisson reconstruction, injection, and back-projection: `src/geocorefusion/fusion.py`
- Dark-region and observation-consistency metrics: `src/geocorefusion/quality.py`
- Manifests and previews: `src/geocorefusion/output.py`
- V6.1 release configurations: `configs/3dssz_roi_fusion_v61.yaml`, `configs/zkh3_roi_fusion_v61.yaml`
- V6.1 coefficient-retaining figure configurations: `configs/3dssz_roi_fusion_v61_benchmark.yaml`, `configs/zkh3_roi_fusion_v61_benchmark.yaml`
- Reproducible benchmark/figure driver: `scripts/build_v61_figures.py`

## Tests

```powershell
python -m pytest -q
```

The suite covers endpoint coordinate scaling, invalid borders, search-boundary tie points, synthetic TRE/EPE behavior, visual-full-detail product contracts, colour-only/equal-luminance edge transfer, and final low-resolution observation control.

## Paper-Stage Validation Priorities

1. Measure held-out TRE on manually annotated or physically calibrated RGB-NIR-SWIR landmarks at the native RGB scale, reporting median, P95, maximum, failure rate, and uncertainty by sensor pair.
2. Add public fusion benchmarks with synthetic sensor degradation and known HR-HSI truth, using ERGAS, SAM, PSNR/SSIM, Q2n/QNR where applicable, and full-resolution consistency metrics.
3. Compare bicubic, CNMF, HySure, LTTR/CSTF, classical GS/PCA/HSV/Brovey/NNDiffuse, recent registration-aware fusion, V5, V6, and V6.1 under identical inputs and registration.
4. Ablate RGB denoising, the four pyramid scales, chroma gradients, Poisson screen strength, dark boost, gain/additive balance, final-product back-projection, and sharpening.
5. Validate on at least three boreholes or independent mining areas, split by borehole/core box, preserving raw DN, calibration, exposure, and acquisition metadata.
6. Add independent physical evidence: high-resolution NIR/SWIR ROIs, repeated scans, point spectroscopy, XRD/Raman, and expert mineral labels.

## Versioning And Rollback

- Current package version: `0.6.1`
- Current method line: `V6.1 denoised RGB visual-full-detail fusion benchmark`
- Current release tag: `v0.6.1-v61`
- Last accepted release tag: `v0.5.0-v5`

Return to this exact V6.1 release with:

```powershell
git checkout v0.6.1-v61
```

The exact accepted V5 release remains available with:

```powershell
git checkout v0.5.0-v5
```
