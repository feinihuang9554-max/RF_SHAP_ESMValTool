import sys
import types

from rf_shap.explain.evaluate import compare_explanations
from rf_shap.namespace import expose_package, install_namespace
from rf_shap.pipeline import DiagnosisPipeline


def test_expose_package_aliases_submodules():
    name = "test_rf_ns_alias"
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
    expose_package(name)
    import test_rf_ns_alias.pipeline as aliased

    assert aliased is sys.modules["rf_shap.pipeline"]
    from test_rf_ns_alias.models.registry import build_model

    assert callable(build_model)


def test_install_namespace_public_api():
    name = "test_rf_ns_install"
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
    install_namespace(pkg)
    assert pkg.DiagnosisPipeline is DiagnosisPipeline
    assert pkg.compare_explanations is compare_explanations
    assert pkg.explain.compare_explanations is compare_explanations
    assert "build_model" in pkg.models.__all__
