"""Phase 0 sanity checks: the foundation imports and is wired correctly.

No application behaviour is tested here — there is none yet.
"""

import importlib

import pytest

ENGINE_PACKAGES = [
    "datapilot",
    "datapilot.config",
    "datapilot.contracts",
    "datapilot.paths",
    "ai_engine",
    "ai_engine.providers",
    "ai_engine.providers.base",
    "backend",
    "data_engine",
    "data_engine.ingestion",
    "data_engine.profiling",
    "data_engine.quality",
    "data_engine.quality.checks",
    "data_engine.cleaning",
    "data_engine.cleaning.rules",
    "data_engine.cleaning.executors",
    "data_engine.preprocessing",
    "data_engine.validation",
    "data_engine.validation.version_models",
    "data_engine.validation.version_store",
    "data_engine.validation.lineage_validation",
    "data_engine.validation.lineage_graph",
    "data_engine.validation.auto_register",
    "data_engine.validation.version_diff",
    "data_engine.validation.integrity",
    "data_engine.validation.store_consistency",
    "data_engine.feature_engineering",
    "database",
    "dl_engine",
    "experimentation",
    "explainability",
    "ml_engine",
]


@pytest.mark.parametrize("module_name", ENGINE_PACKAGES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_version_exposed():
    import datapilot

    assert datapilot.__version__ == "0.0.0"


def test_default_config_loads():
    from datapilot.config import load_config

    cfg = load_config()
    assert cfg["project"]["name"] == "DataPilot"
    assert cfg["determinism"]["global_seed"] == 42


def test_llm_provider_is_abstract():
    from ai_engine.providers.base import LLMProvider

    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]
