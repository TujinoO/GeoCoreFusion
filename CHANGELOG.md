# Changelog

## v0.6.1-v61 - 2026-07-20

- Returned the accepted method line to the direct V6 logic and defined a single `visual_full_detail` product; the V8 `scientific_conditional` gate is not part of the V6.1 run path or product contract.
- Added edge-preserving RGB detail denoising, a four-scale detail pyramid, luminance/chroma strongest-gradient selection, screened-Poisson reconstruction, dark-region relative-contrast boost, and controlled final sharpening.
- Added bounded log-domain multiplicative detail plus a limited additive detail channel so colour-only/equal-luminance RGB edges can enter NIR/SWIR outputs after registration and denoising.
- Added final-product low-resolution back-projection so the actual modulated HR cube, rather than only an intermediate gain field, is cycled toward the measured NIR/SWIR observation grid.
- Added the single-product manifest contract `geocorefusion.visual-full-detail.v1`, explicitly labelling HR pixels as RGB-textured NIR/SWIR estimates and retaining the independent-HR-SWIR non-claim.
- Added 3DSSZ and ZKH3 V6.1 release and coefficient-retaining benchmark configurations, real-run previews, shared-reflectance-domain comparisons, dark/detail/edge audits, spectral-curve reprojection plots, and registration-residual-control figures.
- In the shared-domain benchmark, 2201 nm reliable-detail correlation increased from 0.400 to 0.734 on 3DSSZ and from 0.489 to 0.636 on ZKH3; dark reliable-detail correlation increased from 0.359 to 0.820 and from 0.416 to 0.582, respectively.
- Final-product observation RMSE improved from 0.00462 to 0.00386 on 3DSSZ and from 0.01453 to 0.00946 on ZKH3; SAM improved from 0.923° to 0.872° and from 0.917° to 0.682°.
- Retained the formal boundary that synthetic TRE/EPE demonstrates algorithm-level subpixel capability, while real production registration still requires independent target/landmark TRE before a `<0.5` native-pixel claim.
- Bumped package and runtime-manifest provenance to `0.6.1`; `v0.6.1-v61` is the release tag for this accepted V6.1 state.

## v0.6.0-v6 - 2026-07-20

- Corrected raster endpoint coordinate scaling to map pixel centers with `(N - 1) / (n - 1)` and added regression tests for both endpoints.
- Replaced reflected registration borders with explicit finite-data validity masks so invalid HSI support is not synthesized or scored as evidence.
- Rejected local tie-point peaks on the search boundary and added deterministic TRE/EPE tests for affine and dense residual alignment.
- Added intrinsic log-RGB detail features, local-SNR dark-region confidence, bounded exponential shared gain, and separate gain back-projection.
- Disabled the shared additive-detail path in the V6 configurations and ensured a zero additive map when its strength is zero.
- Added darkest-20% log-high-frequency diagnostics near 901, 1651, and 2201 nm, explicitly labeled as geometry-transfer metrics rather than independent SWIR truth.
- Added matched V5-style controls and V6 configurations for 3DSSZ and ZKH3, plus a reproducible benchmark summary.
- Synthetic known-truth registration reached median/P95 errors of 0.1494/0.2307 pixels for coarse affine TRE, 0.2025/0.2832 pixels for ROI affine TRE, and 0.1850/0.4289 pixels for dense residual EPE on the synthetic analysis grid.
- On matched real ROIs, darkest-20% log-detail correlation increased at all three selected bands while model-space observation-grid self-consistency RMSE/SAM proxies remained at a similar order; these are not strict forward residuals of the final HR cube, and some metrics improved while others regressed slightly.
- Limited the subpixel claim to known synthetic truth. Real-data structure correlations do not establish `<0.5` RGB-pixel registration, and RGB/fused-band detail correlation does not establish true high-resolution NIR/SWIR radiometry.
- Bumped package and runtime-manifest provenance to `0.6.0`; `v0.6.0-v6` remains a tag candidate until the V6 changes are accepted and committed.

## v0.5.0-v5 - 2026-07-19

- Documented the V5 final supplemental method line in `README.md`.
- Marked GeoCoreFusion as package version `0.5.0`.
- Recorded V5 as an observation-constrained material-coefficient detail recovery release.
- Added `/result/` and `/figures/` to `.gitignore` so full ENVI/BigTIFF exports and paper figures stay outside normal Git history.
- Preserved large experiment artifacts as local reproducibility outputs rather than GitHub-tracked files.
