#!/usr/bin/env bash
# Copy the first-PR subset from this overlay into a local clone of
# ESMValGroup/ESMValTool (or your fork).
#
# Scientific kernel is NOT vendored: ESMValTool must depend on / install the
# standalone `rf_shap` package (Manuel / #4609 guidance).
#
#   git clone git@github.com:YOUR_USER/ESMValTool.git ~/src/ESMValTool
#   cd /path/to/RF_SHAP_ESMValTool
#   bash esmvaltool/upstream/prepare_upstream_pr.sh ~/src/ESMValTool
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-}"
if [[ -z "$DEST" || ! -d "$DEST/esmvaltool" ]]; then
  echo "Usage: $0 /path/to/ESMValTool/clone" >&2
  exit 1
fi

mkdir -p "$DEST/esmvaltool/diag_scripts/rf_shap"
mkdir -p "$DEST/esmvaltool/recipes"
mkdir -p "$DEST/doc/sphinx/source/recipes/figures/rf_shap"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '_vendor' \
  "$ROOT/esmvaltool/diag_scripts/rf_shap/" \
  "$DEST/esmvaltool/diag_scripts/rf_shap/"

# Ensure no leftover vendored kernel from older drafts
rm -rf "$DEST/esmvaltool/diag_scripts/rf_shap/_vendor"

cp "$ROOT/esmvaltool/recipes/recipe_rf_shap_demo.yml" "$DEST/esmvaltool/recipes/"
cp "$ROOT/esmvaltool/recipes/recipe_rf_shap_gpp.yml" "$DEST/esmvaltool/recipes/"

cp "$ROOT/esmvaltool/upstream/doc/sphinx/source/recipes/recipe_rf_shap.rst" \
  "$DEST/doc/sphinx/source/recipes/"
cp "$ROOT/esmvaltool/upstream/doc/sphinx/source/recipes/figures/rf_shap/"*.png \
  "$DEST/doc/sphinx/source/recipes/figures/rf_shap/"

# Author tag
python3 - <<PY
from pathlib import Path
path = Path("$DEST") / "esmvaltool" / "config-references.yml"
text = path.read_text()
block = """  huang_feini:
    name: Huang, Feini
    institute: University of Valencia, Spain
    email: feini.huang@uv.es
    orcid:
    github: feinihuang9554-max
"""
anchor = "  hassler_birgit:\n"
if "  huang_feini:\n" not in text:
    if anchor not in text:
        raise SystemExit("could not find hassler_birgit anchor")
    path.write_text(text.replace(anchor, block + anchor, 1))
    print("inserted huang_feini into config-references.yml")
else:
    print("huang_feini already in config-references.yml")
PY

# index.rst entry (idempotent)
INDEX="$DEST/doc/sphinx/source/recipes/index.rst"
if [[ -f "$INDEX" ]] && ! grep -q 'recipe_rf_shap' "$INDEX"; then
  printf '\n   recipe_rf_shap\n' >> "$INDEX"
  echo "appended recipe_rf_shap to recipes/index.rst (please place it in the right toctree section manually)"
fi

cat > "$DEST/esmvaltool/diag_scripts/rf_shap/README.md" <<'EOF'
# rf_shap diagnostic

This diagnostic imports the standalone **`rf_shap`** Python package (scientific
kernel). Do **not** vendor a copy under `_vendor/`.

Install into the same environment as ESMValTool, for example:

```bash
pip install -e /path/to/RF_SHAP_ESMValTool[explain]
```

Development repository:
https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool

Tracked by ESMValTool issue #4609. Packaging as an official ESMValTool /
conda dependency will be finalized with maintainers during review.
EOF

echo
echo "Copied first-PR files into $DEST (diagnostic + recipes + docs; no _vendor)"
echo "Remember: the ESMValTool env must provide the rf_shap package."
echo "Next:"
echo "  cd $DEST"
echo "  git checkout -b diagnostic/rf-shap"
echo "  git add -A && git commit"
echo "  git push -u origin HEAD"
echo "  # open Draft PR mentioning Towards #4609"
