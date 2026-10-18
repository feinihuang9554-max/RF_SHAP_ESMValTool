from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "esmvaltool" / "recipes" / "recipe_rf_shap_demo.yml"
CMIP = ROOT / "esmvaltool" / "recipes" / "recipe_rf_shap.yml"
GPP = ROOT / "esmvaltool" / "recipes" / "recipe_rf_shap_gpp.yml"


def test_demo_recipe_needs_no_datasets():
    recipe = yaml.safe_load(DEMO.read_text())
    assert "datasets" not in recipe
    script = recipe["diagnostics"]["rf_shap"]["scripts"]["rf_shap"]
    assert script["script"] == "rf_shap/diagnostic.py"
    assert script["use_synthetic"] is True
    assert "variables" not in recipe["diagnostics"]["rf_shap"]


def test_cmip_recipe_points_at_diagnostic():
    recipe = yaml.safe_load(CMIP.read_text())
    script = recipe["diagnostics"]["rf_shap"]["scripts"]["rf_shap"]
    assert script["script"] == "rf_shap/diagnostic.py"
    assert script["target"] == "tas"
    assert recipe["datasets"][0]["dataset"] == "ACCESS-CM2"


def test_gpp_recipe_is_cmor_access_esm():
    recipe = yaml.safe_load(GPP.read_text())
    script = recipe["diagnostics"]["rf_shap"]["scripts"]["rf_shap"]
    variables = recipe["diagnostics"]["rf_shap"]["variables"]
    assert script["script"] == "rf_shap/diagnostic.py"
    assert script["target"] == "gpp"
    assert script["predictors"] == ["tas", "hurs", "rsds", "mrsos", "pr"]
    assert variables["gpp"]["mip"] == "Lmon"
    assert recipe["datasets"][0]["dataset"] == "ACCESS-ESM1-5"
    assert "x_files" not in script
    assert script["reference"]["predictor_dataset"] == "ERA5"
    assert script["reference"]["target_dataset"] == "FLUXCOM"
    extra = {
        name: [item["dataset"] for item in spec.get("additional_datasets", [])]
        for name, spec in variables.items()
    }
    assert extra["gpp"] == ["FLUXCOM"]
    assert extra["tas"] == ["ERA5"]
    assert extra["pr"] == ["ERA5"]
