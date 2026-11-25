# rf_shap diagnostic

This diagnostic imports the standalone **`rf_shap`** Python package (scientific
kernel). Do **not** vendor a copy under `_vendor/`.

Install into the same environment as ESMValTool, for example:

```bash
pip install -e /path/to/RF_SHAP_ESMValTool[explain]
# or, for the overlay workflow:
bash esmvaltool/install_links.sh
```

Development repository:
https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool

Tracked by ESMValTool issue #4609. Packaging as an official ESMValTool /
conda dependency will be finalized with maintainers during review.
