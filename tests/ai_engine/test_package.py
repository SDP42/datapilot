"""Phase 11.1-11.4 / 12.1-12.4 — package-level sanity: imports, exports,
and the anthropic-free-import boundary for `ai_engine`.
"""

from __future__ import annotations

import subprocess
import sys

import ai_engine


def test_ai_engine_imports_correctly():
    assert ai_engine is not None
    assert ai_engine.__doc__ is not None


def test_public_exports_are_intentional():
    expected = {
        "DEFAULT_MAX_STEPS",
        "TOOLS_BY_NAME",
        "TOOL_NAMES",
        "AnalysisContext",
        "AnalysisResult",
        "AutonomousRunTrace",
        "ExecutionContext",
        "ExecutionResult",
        "ExecutionStep",
        "LLMMessage",
        "LLMProvider",
        "LLMResponse",
        "Recommendation",
        "RecommendationResult",
        "StepVerdict",
        "ToolSchema",
        "add_section",
        "build_analysis_context",
        "evaluate_step",
        "execute_tool",
        "get_tool_schema",
        "interpret_results",
        "list_tools",
        "recommend_next_steps",
        "render_context_as_text",
        "run_autonomous_experimentation",
    }
    assert set(ai_engine.__all__) == expected
    for name in ai_engine.__all__:
        assert hasattr(ai_engine, name)
    # Phase 11/12 never gives an LLM direct dataframe/model access, never
    # lets a plan branch or run in parallel, and ships exactly one
    # concrete provider so far
    for forbidden in ("openai", "parallel", "branch"):
        assert not any(forbidden in name.lower() for name in ai_engine.__all__)


def test_importing_ai_engine_never_imports_anthropic_at_package_load_time():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ai_engine; import sys; assert 'anthropic' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ai_engine_reuses_existing_training_run_status_not_a_parallel_one():
    # AnalysisResult / RecommendationResult reuse data_engine.modeling's
    # TrainingRunStatus, the same convention dl_engine / experimentation /
    # explainability already follow for their own result contracts.
    from data_engine.modeling import TrainingRunStatus

    assert ai_engine.AnalysisResult.model_fields["status"].annotation is TrainingRunStatus
    assert ai_engine.RecommendationResult.model_fields["status"].annotation is TrainingRunStatus
