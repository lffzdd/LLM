from pathlib import Path

from jsonschema import validators

from ...session import SessionState
from ...skills.registry import SkillRegistry
from ...skills.store import write_skill
from ...skills.types import MAX_ACTIVE_SKILLS
from ...tools.base import ToolRuntime
from ...tools.skill_tools import build_skill_tools
from ...tools.validation import validate_tool_arguments


def _runtime(tmp_path: Path, registry: SkillRegistry | None = None):
    session = SessionState.create("goal", tmp_path)
    tools = build_skill_tools(registry or SkillRegistry(tmp_path))
    runtime = ToolRuntime(workspace_dir=tmp_path, session_state=session)
    return session, tools, runtime


def test_list_load_unload_success_and_failure(tmp_path: Path):
    write_skill(
        tmp_path,
        "release-check",
        name="发布前检查",
        description="发布时使用",
        body="先跑测试",
        allowed_tools=["execute_command"],
    )
    session, tools, runtime = _runtime(tmp_path)
    by_name = {tool.name: tool for tool in tools}

    listed = by_name["list_skills"].call({"query": "发布"}, runtime)
    assert listed.ok
    assert listed.data["count"] == 1
    assert listed.data["skills"][0]["id"] == "release-check"

    loaded = by_name["load_skill"].call({"skill_id": "release-check"}, runtime)
    assert loaded.ok
    assert session.get_active_skill_ids() == ["release-check"]
    assert "先跑测试" in loaded.data["excerpt"]

    missing = by_name["load_skill"].call({"skill_id": "nope"}, runtime)
    assert not missing.ok
    assert "未知 skill" in missing.err

    unloaded = by_name["unload_skill"].call({"skill_id": "release-check"}, runtime)
    assert unloaded.ok
    assert session.get_active_skill_ids() == []

    not_active = by_name["unload_skill"].call({"skill_id": "release-check"}, runtime)
    assert not not_active.ok
    assert "未激活" in not_active.err


def test_activation_limits_are_rejected_with_clear_errors(tmp_path: Path):
    for index in range(MAX_ACTIVE_SKILLS + 1):
        write_skill(
            tmp_path,
            f"skill-{index}",
            name=f"S{index}",
            description=f"desc {index}",
            body="x" * 10,
        )
    session, tools, runtime = _runtime(tmp_path)
    load = {tool.name: tool for tool in tools}["load_skill"]
    for index in range(MAX_ACTIVE_SKILLS):
        result = load.call({"skill_id": f"skill-{index}"}, runtime)
        assert result.ok
    overflow = load.call({"skill_id": f"skill-{MAX_ACTIVE_SKILLS}"}, runtime)
    assert not overflow.ok
    assert str(MAX_ACTIVE_SKILLS) in overflow.err
    assert session.get_active_skill_ids() == [
        f"skill-{index}" for index in range(MAX_ACTIVE_SKILLS)
    ]


def test_body_budget_is_rejected(tmp_path: Path):
    write_skill(
        tmp_path,
        "first",
        name="一",
        description="先加载",
        body="a" * 8000,
    )
    write_skill(
        tmp_path,
        "second",
        name="二",
        description="会超限",
        body="b" * 8000,
    )
    session, tools, runtime = _runtime(tmp_path)
    load = {tool.name: tool for tool in tools}["load_skill"]
    assert load.call({"skill_id": "first"}, runtime).ok
    overflow = load.call({"skill_id": "second"}, runtime)
    assert not overflow.ok
    assert "超过上限" in overflow.err
    assert session.get_active_skill_ids() == ["first"]


def test_skill_tool_schemas_are_valid():
    tools = build_skill_tools(SkillRegistry(Path(".")))
    for tool in tools:
        validator_cls = validators.validator_for(tool.parameters)
        validator_cls.check_schema(tool.parameters)
        if tool.name == "load_skill":
            assert validate_tool_arguments(tool, {"skill_id": "ok"}) is None
            invalid = validate_tool_arguments(tool, {"skill_id": "../x"})
            assert invalid is not None
            assert invalid.data["error"]["type"] == "tool_input_validation"
