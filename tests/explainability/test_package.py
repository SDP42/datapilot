"""Phase 10.1 / 10.2 / 10.3 / 10.4 — package-level sanity: imports,
exports, and the shap-free-import boundary for `explainability`.
"""

from __future__ import annotations

import subprocess
import sys

import explainability


def test_explainability_imports_correctly():
    assert explainability is not None
    assert explainability.__doc__ is not None


def test_public_exports_are_intentional():
    expected = {
        "ExplainabilityStatus",
        "ExplanationMethod",
        "ExplanationReport",
        "ExplanationRequest",
        "FeatureImportanceEntry",
        "FeatureImportanceResult",
        "PartialDependencePoint",
        "PartialDependenceResult",
        "SHAPAvailability",
        "compute_partial_dependence",
        "compute_permutation_importance",
        "compute_shap_importance",
        "is_shap_available",
        "shap_availability",
        "understand_explanation",
    }
    assert set(explainability.__all__) == expected
    for name in explainability.__all__:
        assert hasattr(explainability, name)
    # Phase 10 foundation-through-10.4: no model fitting here, no composed
    # pipeline, no LIME, no counterfactuals, no natural-language text
    for forbidden in ("fit", "pipeline", "lime", "counterfactual", "recommend"):
        assert not any(forbidden in name.lower() for name in explainability.__all__)


def test_importing_explainability_never_imports_shap_at_package_load_time():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import explainability; import sys; assert 'shap' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
