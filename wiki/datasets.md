# Datasets — Phase 2

**Same AvalCD geographic split as Phase 1 + pixel-level GT rasters for segmentation + Tromsø D-scale annotations + SeNorge snowpack variables.**

---

## AvalCD — primary dataset

- **Citation**: Gatti, M. et al. (2026). *AvalCD dataset.* Zenodo. doi:[10.5281/zenodo.15863589](https://zenodo.org/records/15863589)
- **License**: CC BY-NC 4.0
- Source: Zenodo (1.1 GB)
- 4 regions: Livigno (Italy), Nuuk (Greenland), Pish (Tajikistan), Tromsø (Norway)
- 7 acquisition events
- Phase 2 uses **pixel-level GT rasters** (not just patch labels) — all scenes have binary debris masks

> ⚠ OPEN: Zenodo lists the dataset creator as "Mattia Gatti (University of Insubria)" but the
> companion paper (arXiv:2603.22658) is attributed to "Gattimgatti et al." Verify whether these
> are the same person (surname variation) before final paper submission. Phase 1 README currently
> cites the dataset as "Gattimgatti et al." which may be wrong.

### Geographic split (identical to Phase 1 and Gattimgatti 2026)

| Split | Scenes | Notes |
|-------|--------|-------|
| Train | Livigno_20240403, Livigno_20250129, Nuuk_20160413, Nuuk_20210411, Pish_20230221 | |
| Val | Livigno_20250318 | |
| Test OOD | Tromsø_20241220 | **never seen in training; only scene with D-scale labels** |

---

## Tromsø GT — size annotations

- 117 debris polygons with D-scale labels and area measurements
- Distribution: 5 × D1, 25 × D2, 71 × D3, 16 × D4
- `size` column: int {1, 2, 3, 4}
- `area` column: float, m²
- Used for: D-scale proxy calibration, size estimation experiment, D2 detection analysis

> ⚠ OPEN: D1 (n=5) is likely below Sentinel-1 detection floor. May need to exclude from size classifier or treat as "undetectable" class.

---

## SeNorge — snowpack variables (Tromsø supplementary)

- Norwegian national 1km daily snow model
- Variables of interest:
  - `ΔSD₇₂`: 72-hour snow depth change (m) — proxy for fresh snow load
  - bulk density (kg/m³)
- Date: 2024-12-19 to 2024-12-20 (day before and day of Tromsø acquisition)
- Access: thredds.met.no (NetCDF, open access)
- Spatial resolution: 1 km — must be resampled/interpolated to Sentinel-1 10m grid
- Used only for: snowpack fusion experiment on Tromsø (supplementary goal)

> ⚠ OPEN: SeNorge download not yet done. Need to identify exact variable names in the NetCDF schema and confirm December 2024 availability.

---

## Input channels (Phase 2 — expanded for small deposit detection)

Phase 1 used 5 channels. Phase 2 adds log-ratio, LIA-normalized, and cross-pol ratio channels:

| Channel | Description | New? |
|---------|-------------|------|
| VH_post (dB) | Post-event VH backscatter, clipped [−25,−5] dB | Phase 1 |
| VV_post (dB) | Post-event VV backscatter, clipped [−25,−5] dB | Phase 1 |
| slope | Terrain slope from DEM | Phase 1 |
| sin(aspect) | Terrain aspect (sine component) | Phase 1 |
| cos(aspect) | Terrain aspect (cosine component) | Phase 1 |
| VH_pre (dB) | Pre-event VH backscatter | Phase 1 (bi-temporal) |
| VV_pre (dB) | Pre-event VV backscatter | Phase 1 (bi-temporal) |
| log(VH_post/VH_pre) | Log-ratio change, VH — multiplicative speckle suppression | **Phase 2** |
| log(VV_post/VV_pre) | Log-ratio change, VV | **Phase 2** |
| VH_post/VV_post | Cross-pol ratio, post — surface roughness proxy | **Phase 2** |
| VH_pre/VV_pre | Cross-pol ratio, pre | **Phase 2** |
| LIA | Local Incidence Angle (raster already in AvalCD) | **Phase 2** |

All SAR channels (VH, VV) LIA-normalized before log-ratio and ratio computation.

> ⚠ OPEN: final channel count is 12. Norm stats from Phase 1 (7 channels) are invalid —
> new norm stats must be computed on train split before any training run.

> ✓ DECIDED: backbone will be retrained from scratch with 12-channel input. Phase 1 checkpoint
> is not reused. New norm stats must be computed on train split before any training run.

---

## Normalization

- Phase 1 norm stats (`norm_stats_bitemporal.json`) are **not reusable** — channel set changed.
- New norm stats must be computed on train split (Livigno+Nuuk+Pish) before training.

---

## What's new in Phase 2

- Pixel-level GT rasters (already in AvalCD; Phase 1 didn't use them)
- Expanded input channels: log-ratio, cross-pol ratio, LIA normalization
- SeNorge variables (new download, supplementary)
- No new SAR scenes added
