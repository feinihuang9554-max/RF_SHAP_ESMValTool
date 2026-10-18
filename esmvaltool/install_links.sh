#!/usr/bin/env bash
# Link this repository's diagnostic and recipe into an ESMValTool install
# and install rf_shap into the same Python environment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ESMVALTOOL_PYTHON:-$(command -v python)}"
if ! "$PYTHON" -c "import esmvaltool" >/dev/null 2>&1; then
  echo "This Python cannot import esmvaltool: $PYTHON" >&2
  echo "Activate the esmvaltool conda env, or set ESMVALTOOL_PYTHON to that interpreter." >&2
  exit 1
fi
SITE="$("$PYTHON" -c "import esmvaltool, pathlib; print(pathlib.Path(esmvaltool.__file__).parent)")"

mkdir -p "$SITE/diag_scripts" "$SITE/recipes"
ln -sfn "$ROOT/esmvaltool/diag_scripts/rf_shap" "$SITE/diag_scripts/rf_shap"
ln -sfn "$ROOT/esmvaltool/python_api" "$SITE/rf_shap"
ln -sfn "$ROOT/esmvaltool/recipes/recipe_rf_shap.yml" "$SITE/recipes/recipe_rf_shap.yml"
ln -sfn "$ROOT/esmvaltool/recipes/recipe_rf_shap_demo.yml" "$SITE/recipes/recipe_rf_shap_demo.yml"
ln -sfn "$ROOT/esmvaltool/recipes/recipe_rf_shap_gpp.yml" "$SITE/recipes/recipe_rf_shap_gpp.yml"
ln -sfn "$ROOT/esmvaltool/recipes/recipe_rf_shap_cloud.yml" "$SITE/recipes/recipe_rf_shap_cloud.yml"

"$PYTHON" -m pip install -e "$ROOT[explain]"

HOOK="$SITE/__init__.py"
if ! grep -q "rf_shap API hook" "$HOOK"; then
  cat >> "$HOOK" << 'PY'

# rf_shap API hook (added by RF_SHAP_ESMValTool install_links.sh)
def __getattr__(name):
    if name == "rf_shap":
        from importlib import import_module
        return import_module("esmvaltool.rf_shap")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
PY
fi

echo "Linked diagnostic -> $SITE/diag_scripts/rf_shap"
echo "Linked Python API  -> $SITE/rf_shap  (import esmvaltool.rf_shap)"
echo "Linked recipes     -> $SITE/recipes/recipe_rf_shap.yml"
echo "                     $SITE/recipes/recipe_rf_shap_demo.yml"
echo "                     $SITE/recipes/recipe_rf_shap_gpp.yml"
echo "                     $SITE/recipes/recipe_rf_shap_cloud.yml"
echo "Smoke test (no climate files): esmvaltool run recipe_rf_shap_demo.yml"
echo "CMIP tas example:              esmvaltool run recipe_rf_shap.yml"
echo "CMIP GPP vs FluxCom/ERA5:      esmvaltool run recipe_rf_shap_gpp.yml"
echo "CMIP clt vs ERA5:              esmvaltool run recipe_rf_shap_cloud.yml"
echo "If recipes are not found by name, pass the path under $ROOT/esmvaltool/recipes/"
