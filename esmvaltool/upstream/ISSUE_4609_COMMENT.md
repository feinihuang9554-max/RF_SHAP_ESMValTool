## Status update (RF-SHAP / AI4PEX WP4)

Development repository (standalone `rf_shap` package + ESMValTool 2.15 overlay):
https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool

### What already works locally
- `recipe_rf_shap_demo.yml` — synthetic ExtraTrees + SHAP smoke test (**no climate files**)
- Stock ESMValTool path verified on the same install (local ACCESS-CM2 `tas` + official `examples/diagnostic.py`)
- Domain recipes prepared in the overlay: GPP/land, cloud, ocean
- Python API importable as `esmvaltool.rf_shap` after `bash esmvaltool/install_links.sh`
- Scientific kernel installed as an editable dependency (`pip install -e ".[explain]"`), **not** vendored under `_vendor/`

### Packaging choice (aligned with maintainer feedback)
Having a **standalone `rf_shap` package** that ESMValTool imports still counts as
“run in ESMValTool”. The first upstream PR will ship the diagnostic scripts and
recipes, and declare / install `rf_shap` as a dependency (no `_vendor/rf_shap` tree).

### Proposed first upstream PR (minimal)
Towards #4609, first PR scope:
1. `esmvaltool/diag_scripts/rf_shap/` (diagnostic + compare; imports installed `rf_shap`)
2. `recipe_rf_shap_demo.yml` (always runnable)
3. `recipe_rf_shap_gpp.yml` (land / M4.4-aligned scientific example)
4. Sphinx recipe page + example figures
5. Author tag `huang_feini` in `config-references.yml`

Cloud (`recipe_rf_shap_cloud.yml`) and ocean (`recipe_rf_shap_ocean.yml`) stay in follow-up PRs.

### Remaining before Ready-for-review
- [x] Copy diagnostic/recipes into an `ESMValGroup/ESMValTool` fork branch (Draft PR #4658)
- [x] Drop vendored `_vendor/rf_shap`; use standalone package dependency instead
- [ ] Sign CLA on first PR
- [ ] Confirm maintainer tag (proposal: `swaminathan_ranjini`; author: `huang_feini`)
- [ ] Agree with maintainers how `rf_shap` enters the ESMValTool env (pip/conda)
- [ ] Dependency handling for optional `shap` / `fasttreeshap`
- [ ] Tighten provenance ancestors for diagnostic outputs

### Owners / timeline
- Implementation & draft PR: Feini Huang (`@feinihuang9554-max`)
- Issue / WP4 contact: `@rswamina`
- Target: keep Draft PR updated; Ready-for-review after CI/docs/CLA green and dependency packaging agreed

Upstream packaging notes and RST draft live in the development repo under `esmvaltool/upstream/`.
