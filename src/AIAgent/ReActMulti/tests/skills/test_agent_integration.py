import json
from pathlib import Path

from ...agent import Agent
from ...checkpoint import SessionCheckpointStore
from ...events import ContentDone
from ...renderer import SilentRenderer
from ...session import SessionState
from ...skills.registry import SkillRegistry
from ...skills.store import write_skill
from ...tools.skill_tools import build_skill_tools


class ScriptLLM:
    context_limit = 128_000

    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []

    def __call__(self, messages):
        self.seen_messages.append(list(messages))
        yield ContentDone(content=self.script.pop(0))


def _tool(name, **arguments):
    return json.dumps({
        "tool_calls": [{"name": name, "arguments": arguments}],
        "final_answer": None,
    }, ensure_ascii=False)


def _final(answer):
    return json.dumps({"tool_calls": [], "final_answer": answer}, ensure_ascii=False)


def _write_skill(directory: Path) -> SkillRegistry:
    write_skill(
        directory,
        "release-check",
        name="发布前检查",
        description="发布时使用的检查流程",
        body="发布前必须先跑测试。",
        allowed_tools=["execute_command"],
    )
    return SkillRegistry(directory)


def test_parent_and_child_sessions_isolate_activation(tmp_path: Path):
    registry = _write_skill(tmp_path)
    parent = SessionState.create("parent", tmp_path)
    child = SessionState.create("child", tmp_path)
    parent.activate_skill("release-check")
    assert parent.get_active_skill_ids() == ["release-check"]
    assert child.get_active_skill_ids() == []
    assert registry.get("release-check").id == "release-check"


def test_empty_skills_directory_injects_nothing(tmp_path: Path):
    llm = ScriptLLM([_final("done")])
    session = SessionState.create("task", tmp_path)
    Agent(llm, [], session, SilentRenderer(), skills=SkillRegistry(tmp_path)).run("hi")
    assert not any(
        "<skill-catalog>" in str(message.get("content", ""))
        or "<skill id=" in str(message.get("content", ""))
        for batch in llm.seen_messages
        for message in batch
    )
    assert not any(
        "<system-reminder>" in str(record.message.get("content", ""))
        for record in session.message_records
    )


def test_catalog_and_body_are_ephemeral(tmp_path: Path):
    registry = _write_skill(tmp_path)
    llm = ScriptLLM([
        _tool("load_skill", skill_id="release-check"),
        _final("已按流程检查"),
    ])
    session = SessionState.create("task", tmp_path)
    tools = build_skill_tools(registry)
    answer = Agent(
        llm, tools, session, SilentRenderer(), skills=registry
    ).run("准备发布")

    assert answer == "已按流程检查"
    assert "<skill-catalog>" in llm.seen_messages[0][-1]["content"]
    assert "release-check" in llm.seen_messages[0][-1]["content"]
    assert "<skill id=" not in llm.seen_messages[0][-1]["content"]

    assert "<skill id=\"release-check\"" in llm.seen_messages[1][-1]["content"]
    assert "发布前必须先跑测试" in llm.seen_messages[1][-1]["content"]

    transcript = [
        str(record.message.get("content", ""))
        for record in session.message_records
    ]
    assert all("<skill-catalog>" not in text for text in transcript)
    assert all("<skill id=" not in text for text in transcript)


def test_unload_removes_body_from_next_turn(tmp_path: Path):
    registry = _write_skill(tmp_path)
    llm = ScriptLLM([
        _tool("load_skill", skill_id="release-check"),
        _tool("unload_skill", skill_id="release-check"),
        _final("已卸载"),
    ])
    session = SessionState.create("task", tmp_path)
    Agent(
        llm,
        build_skill_tools(registry),
        session,
        SilentRenderer(),
        skills=registry,
    ).run("准备发布")

    assert "<skill id=\"release-check\"" in llm.seen_messages[1][-1]["content"]
    assert "<skill id=\"release-check\"" not in llm.seen_messages[2][-1]["content"]
    assert session.get_active_skill_ids() == []


def test_new_user_turn_resets_activation(tmp_path: Path):
    registry = _write_skill(tmp_path)
    llm = ScriptLLM([
        _tool("load_skill", skill_id="release-check"),
        _final("第一轮完成"),
        _final("第二轮完成"),
    ])
    session = SessionState.create("task", tmp_path)
    agent = Agent(
        llm,
        build_skill_tools(registry),
        session,
        SilentRenderer(),
        skills=registry,
    )
    assert agent.run("发布") == "第一轮完成"
    assert session.get_active_skill_ids() == ["release-check"]
    assert agent.run("另一件事") == "第二轮完成"
    assert session.get_active_skill_ids() == []
    assert "<skill id=\"release-check\"" not in llm.seen_messages[2][-1]["content"]


def test_continue_run_keeps_activation(tmp_path: Path):
    registry = _write_skill(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionCheckpointStore(tmp_path / "checkpoints")
    session = SessionState.create("task", workspace)
    session.begin_user_turn("准备发布")
    session.append_message({"role": "user", "content": "准备发布"})
    session.activate_skill("release-check")
    store.save(session)

    restored = store.load(session.session_id)
    assert restored.get_active_skill_ids() == ["release-check"]
    llm = ScriptLLM([_final("继续")])
    answer = Agent(
        llm, [], restored, SilentRenderer(), skills=registry
    ).continue_run()
    assert answer == "继续"
    assert restored.get_active_skill_ids() == ["release-check"]
    assert "发布前必须先跑测试" in llm.seen_messages[0][-1]["content"]
    assert all(
        "<skill id=" not in str(record.message.get("content", ""))
        for record in restored.message_records
    )
