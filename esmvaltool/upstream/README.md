# Upstream packaging for ESMValTool issue #4609

This folder prepares the **first official pull request** into
[`ESMValGroup/ESMValTool`](https://github.com/ESMValGroup/ESMValTool)
tracked by [#4609](https://github.com/ESMValGroup/ESMValTool/issues/4609).

The live development workflow remains the symlink overlay
(`bash esmvaltool/install_links.sh`) plus editable install of the standalone
`rf_shap` package. Upstream gets **diagnostic + recipes + docs only** — the
scientific kernel stays a **dependency**, not a vendored `_vendor/rf_shap` tree
(agreed with Manuel: standalone package imported by ESMValTool still counts as
“run in ESMValTool”).

## First PR contents

| Path | Role |
| --- | --- |
| `diag_scripts/rf_shap/` | Diagnostic + compare (imports installed `rf_shap`) |
| `recipes/recipe_rf_shap_demo.yml` | Always-on smoke recipe |
| `recipes/recipe_rf_shap_gpp.yml` | Land / GPP scientific example |
| `doc/.../recipe_rf_shap.rst` + figures | User-guide page |
| `config-references.yml` author `huang_feini` | Recipe author tag |

Cloud and ocean recipes stay out of the first PR.

**Not in the PR:** a copy of `src/rf_shap`. Install / depend on the package
instead (`pip install -e …[explain]` during review; conda/PyPI packaging with
maintainers later).

## Steps

1. **Local author tag** (so overlay recipes keep validating):

   ```bash
   bash esmvaltool/upstream/apply_local_author_tag.sh
   ```

2. **Comment on #4609** — paste
   [`ISSUE_4609_COMMENT.md`](ISSUE_4609_COMMENT.md)
   (needs a working `gh` login or the GitHub web UI).

3. **Fork** `ESMValGroup/ESMValTool` on GitHub, then:

   ```bash
   git clone git@github.com:feinihuang9554-max/ESMValTool.git ~/src/ESMValTool
   cd /path/to/RF_SHAP_ESMValTool
   bash esmvaltool/upstream/prepare_upstream_pr.sh ~/src/ESMValTool
   cd ~/src/ESMValTool
   git checkout -b diagnostic/rf-shap
   git add -A
   git commit -m "Add rf_shap diagnostic (AI4PEX / Towards #4609)"
   git push -u origin diagnostic/rf-shap
   ```

4. Open a **Draft** pull request against `ESMValGroup/ESMValTool`:
   - title: `Add rf_shap ExtraTrees+SHAP diagnostic (Towards #4609)`
   - sign the CLA when prompted
   - do **not** request review until CI/Codacy/docs are green and
     maintainer packaging of `rf_shap` as a dependency is agreed

## Author / maintainer tags used in recipes

- authors: `huang_feini`
- maintainer: `swaminathan_ranjini` (issue reporter; confirm if needed)
