import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prepare_cmor_names",
    Path(__file__).resolve().parents[1] / "esmvaltool" / "prepare_cmor_names.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
cmor_filename = _MOD.cmor_filename
link_tree = _MOD.link_tree


def test_cmor_filename_inserts_grid():
    path = Path("tas_Amon_ACCESS-CM2_historical_r1i1p1f1_198001-201412.nc")
    assert (
        cmor_filename(path, "gn")
        == "tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_198001-201412.nc"
    )


def test_cmor_filename_keeps_existing_grid():
    path = Path("tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_198001-201412.nc")
    assert cmor_filename(path, "gn") == path.name


def test_link_tree(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "huss_Amon_ACCESS-CM2_historical_r1i1p1f1_198001-201412.nc").write_text("x")
    dst = tmp_path / "dst"
    created = link_tree(src, dst, "gn")
    assert len(created) == 1
    assert created[0].name == "huss_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_198001-201412.nc"
    assert created[0].resolve() == (src / "huss_Amon_ACCESS-CM2_historical_r1i1p1f1_198001-201412.nc").resolve()
