"""Phase 11.2 — the fixed tool-schema vocabulary (`ai_engine.tools`)."""

from __future__ import annotations

from ai_engine.tools import TOOL_NAMES, TOOLS_BY_NAME, get_tool_schema, list_tools


def test_tool_names_is_non_empty_and_matches_tools_by_name():
    assert len(TOOL_NAMES) > 0
    assert TOOL_NAMES == frozenset(TOOLS_BY_NAME)


def test_every_tool_has_a_valid_json_schema_object():
    for tool in list_tools():
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters
        assert isinstance(tool.description, str) and tool.description


def test_get_tool_schema_returns_matching_tool():
    tool = get_tool_schema("analyze_quality")
    assert tool.name == "analyze_quality"


def test_get_tool_schema_unknown_name_raises_key_error():
    import pytest

    with pytest.raises(KeyError):
        get_tool_schema("not_a_real_tool")


def test_list_tools_is_stable_order():
    assert [t.name for t in list_tools()] == [t.name for t in list_tools()]


def test_tool_names_are_unique():
    names = [t.name for t in list_tools()]
    assert len(names) == len(set(names))
