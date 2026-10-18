"""``import esmvaltool.rf_shap`` — same API as the ``rf_shap`` package."""

from rf_shap.namespace import install_namespace
import sys

install_namespace(sys.modules[__name__])
