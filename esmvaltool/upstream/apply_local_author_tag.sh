#!/usr/bin/env bash
# Insert huang_feini into the active ESMValTool config-references.yml so local
# overlay recipes validate before the upstream PR merges.
set -euo pipefail

PYTHON="${ESMVALTOOL_PYTHON:-/opt/miniconda3/envs/esmvaltool/bin/python}"
REF="$("$PYTHON" - <<'PY'
import esmvaltool
from pathlib import Path
print(Path(esmvaltool.__file__).resolve().parent / "config-references.yml")
PY
)"

if [[ ! -f "$REF" ]]; then
  echo "config-references.yml not found via $PYTHON" >&2
  exit 1
fi

if grep -q '^  huang_feini:' "$REF"; then
  echo "huang_feini already present in $REF"
  exit 0
fi

python3 - <<PY
from pathlib import Path
path = Path("$REF")
text = path.read_text()
needle = "  hardacre_catherine:\n"
block = """  huang_feini:
    name: Huang, Feini
    institute: University of Valencia, Spain
    email: feini.huang@uv.es
    orcid:
    github: feinihuang9554-max
"""
if needle not in text:
    raise SystemExit(f"insert anchor not found in {path}")
# place after hardacre block's orcid line by inserting before hassler
anchor = "  hassler_birgit:\n"
if anchor not in text:
    raise SystemExit(f"hassler_birgit anchor not found in {path}")
if "  huang_feini:\n" in text:
    print("already present")
else:
    text = text.replace(anchor, block + anchor, 1)
    path.write_text(text)
    print(f"inserted huang_feini into {path}")
PY
