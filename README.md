# GeoCoreFusion

**Version:** V5 final supplemental result, package `0.5.0`, Git tag `v0.5.0-v5`

GeoCoreFusion is a research prototype for close-range drill-core RGB-NIR-SWIR image fusion. The current V5 line targets a specific scientific problem: how to register heterogeneous push-broom sensors, harmonize NIR/SWIR spectra, and recover RGB-guided spatial detail while keeping the fused cube consistent with the original NIR/SWIR observations.

The temporary paper title used in the V5 research document is:

> GeoCoreFusion: Physics-Constrained Self-Supervised Registration, Spectral Harmonization and Detail-Preserving Fusion for RGB-NIR-SWIR Drill-Core Imagery.

## Scope

The fused continuous hyperspectral product is defined on the RGB spatial grid over **691-2518 nm**, sampled at about **5 nm** for **367 bands**. RGB is treated as a high-spatial-resolution structural reference, not as hyperspectral ground truth below 691 nm and not as a template copied into every SWIR band.

Current V5 results are self-supervised and observation-consistency based. They support technical feasibility and staged paper preparation, but they do not replace future independent validation by high-resolution NIR/SWIR ROI, point spectroscopy, repeated scans, XRD/Raman, or expert mineral labels.

## V5 Method

V5 is organized into four linked stages.

1. **Automatic local registration:** after global ECC/affine initialization, the pipeline builds cross-modal structure maps on a low-resolution observation grid, generates local control-point candidates, filters them by forward-backward consistency, matching score, peak margin, MAD displacement, and maximum displacement, then interpolates smooth local displacement fields with confidence-weighted IDW.
2. **NIR/SWIR spectral harmonization:** the NIR and SWIR overlap around **978-1482 nm** is used to estimate wavelength-dependent gain and offset. The final axis spans **691-2518 nm**; low-SNR water absorption regions near **1400 nm** and **1900 nm** are recorded as low-confidence bands in `band_metadata.csv`.
3. **Low-rank material coefficient representation:** the low-resolution unified cube is represented as `X_LR = mu + A_LR E`, where `E` is learned only from NIR/SWIR observations and `A_LR` is the material/coefficient field. Fusion estimates high-resolution `A_HR`, then reconstructs `X_HR = mu + A_HR E`.
4. **Observation-constrained spatial detail recovery:** RGB is converted into luminance, red-green difference, and blue-green difference structure features. V5 performs signed coefficient-level detail regression, material-boundary and confidence gating, observation-near-nullspace projection, low-resolution back-projection, and a conservative mixture of weak multiplicative gain plus band-adaptive additive detail.

This means V5 is not simple sharpening. Its central design is **material-coefficient detail recovery constrained by the sensor observation model**.

## Current V5 Results

Recommended V5 candidates from the supplied V5 final-result document:

| Dataset | Recommended V5 candidate | Coefficient detail | Multiplicative gain | Additive detail | Use case |
|---|---:|---:|---:|---:|---|
| 3DSSZ | Structure-strong v5 | 0.60 | 0.13 | 0.26 | Stronger structure for rich rock boundaries and internal texture |
| ZKH3 | Hybrid-sharp v5 | 0.26 | 0.26 | 0.18 | More stable balance between visual clarity and observation consistency |

V4 to V5 quantitative comparison:

| Dataset | Coefficient RMSE | Continuous-cube RMSE | SAM | 900 nm high-frequency energy |
|---|---:|---:|---:|---:|
| 3DSSZ | 0.00753 -> 0.00665 | 0.00437 -> 0.00449 | 0.9076 deg -> 0.9050 deg | 0.0863 -> 0.0986 |
| ZKH3 | 0.04903 -> 0.04050 | 0.01017 -> 0.00909 | 0.6425 deg -> 0.5570 deg | 0.0763 -> 0.0978 |

Observation-consistency summary after degrading the V5 high-resolution cube back to the sensor observation grid:

| Dataset | Continuous-cube observation RMSE | Mean SAM | Mean per-band spatial correlation | Interpretation |
|---|---:|---:|---:|---|
| 3DSSZ | 0.00449 | 0.905 deg | 0.99958 | High consistency remains under strong absorption and complex spectra |
| ZKH3 | 0.00909 | 0.557 deg | 0.99830 | Main absorption positions and overall spectral shape remain stable |

Representative ROI checks in the V5 document report single-ROI RMSE of about **0.0024-0.0041** and SAM of about **0.21-0.50 deg** for 3DSSZ, and RMSE of about **0.0019-0.0030** and SAM of about **0.19-0.34 deg** for ZKH3.

## Repository Contents

```text
configs/                  V3/V4/V5 ROI configuration files
docs/                     Method notes, registration review notes, V5 review notes
scripts/                  Diagnostic and output-validation helpers
src/geocorefusion/         Python package implementation
tests/                    Unit and synthetic pipeline tests
```

Large local experiment artifacts are intentionally excluded from Git:

- `data/` raw RGB/NIR/SWIR ENVI observations
- `runs/` intermediate and ROI run outputs
- `result/` full V5 ENVI/BigTIFF exports, about 2.2 GiB per full cube
- `figures/` paper/review figure exports

This keeps GitHub useful for code versioning and rollback. Store full data products through a release asset, data repository, object storage, or project archive instead of normal Git history.

## Install

```powershell
conda env create -f environment.yml
conda activate geocorefusion
python -m pip install -e .
```

If the current Anaconda environment already has NumPy, SciPy, OpenCV-contrib, Pillow, PyYAML, and scikit-learn:

```powershell
python -m pip install -e . --no-deps
```

## Run V5

Inspect source data:

```powershell
geocorefusion inspect "E:\Code\GeoCoreFusion\data\2023_09_04_14_34_41-3DSSZ-1-16-267.0_276.0-532.0_552.0"
```

Run the finalized V5 ROI workflows:

```powershell
python -m geocorefusion.cli run configs\3dssz_roi_fusion_v5.yaml
python -m geocorefusion.cli run configs\zkh3_roi_fusion_v5.yaml
```

Each complete 367-band V5 output is about **2.3 GiB**, so full-cube regeneration should be done after comparison and ablation parameters are fixed.

Validate an output directory:

```powershell
geocorefusion validate "E:\Code\GeoCoreFusion\runs\3dssz_roi_fusion_v5"
```

## Output Layout

```text
runs/<run>/
├─ cube/fused_continuous_691_2518nm.hdr|dat
├─ cube/fused_691_2518nm_envi.hdr|img
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

## Implementation Map

- Registration and local tie points: `src/geocorefusion/registration.py`, `src/geocorefusion/pipeline.py`
- Spectral harmonization: `src/geocorefusion/spectral.py`
- V5 coefficient detail recovery and fusion: `src/geocorefusion/fusion.py`
- Preview and ENVI output writing: `src/geocorefusion/output.py`
- Quality evaluation: `src/geocorefusion/quality.py`
- V5 configurations: `configs/3dssz_roi_fusion_v5.yaml`, `configs/zkh3_roi_fusion_v5.yaml`
- V5 review note: `docs/v5空间细节恢复与人工复核.md`
- Technical route note: `docs/技术路线实现.md`

## Tests

```powershell
python -m pytest -q
```

## Paper-Stage Experiment Plan

The supplied V5 document recommends the next stage move from "visually better" to "provable." Priority experiments:

- Baselines: bicubic upsampling, CNMF, HySure, LTTR/CSTF, project V3/V4/V5, legacy M0-1 classical, and reproducible Transformer/Unfolding/diffusion baselines after 2022.
- Public datasets: CAVE, Harvard, Chikusei, and Pavia for general fusion behavior.
- Ablations: no local control-point registration, no NIR-SWIR overlap harmonization, V4 multiplicative gain versus V5 coefficient detail regression, no material-boundary/confidence gating, no near-nullspace projection, no low-resolution back-projection, and no band-adaptive additive detail.
- Real drill-core validation: at least 3 boreholes or independent mining areas, 30-50 core boxes, split by borehole/core box, with preserved DN, calibration and acquisition metadata.
- Independent evidence: repeated scans, point spectroscopy, high-resolution NIR/SWIR ROI, XRD/Raman, expert mineral labels, and manual scoring for registration residuals and spectral splice artifacts.

## Versioning And Rollback

This repository uses the package version plus Git tags:

- Current package version: `0.5.0`
- Current method release: `V5 final supplemental result`
- Current tag: `v0.5.0-v5`

After pulling the repository in the future, return to this exact version with:

```powershell
git checkout v0.5.0-v5
```

To continue development after checking out a tag:

```powershell
git switch main
```
