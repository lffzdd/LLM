"""把 SkillRegistry 暴露成 list/load/unload 工具。不给模型写 skill 的能力。"""

from __future__ import annotations

from ..skills.registry import SkillRegistry
from ..skills.types import (
    MAX_ACTIVE_BODY_CHARS,
    SkillActivationError,
    SkillNotFoundError,
    SkillStoreError,
)
from .base import Tool, ToolResult, ToolRuntime


def _session(runtime: ToolRuntime | None):
    session = runtime.session_state if runtime is not None else None
    if session is None:
        raise RuntimeError("skill 工具需要 SessionState")
    return session


def _active_ids(session) -> list[str]:
    getter = getattr(session, "get_active_skill_ids", None)
    if getter is not None:
        return list(getter())
    return list(getattr(session, "active_skill_ids", []))


def list_skills(
    query: str = "",
    runtime: ToolRuntime | None = None,
    *,
    registry: SkillRegistry,
) -> ToolResult:
    del runtime
    try:
        metas = registry.list_metas(query)
    except SkillStoreError as exc:
        return ToolResult.fail(str(exc))
    return ToolResult.success({
        "count": len(metas),
        "skills": [
            {
                "id": meta.id,
                "name": meta.name,
                "description": meta.description,
            }
            for meta in metas
        ],
    })


def load_skill(
    skill_id: str,
    runtime: ToolRuntime | None = None,
    *,
    registry: SkillRegistry,
) -> ToolResult:
    session = _session(runtime)
    try:
        definition = registry.get(skill_id)
    except (SkillNotFoundError, SkillStoreError) as exc:
        return ToolResult.fail(str(exc))

    active = _active_ids(session)
    if definition.id in active:
        return ToolResult.success({
            "message": f"skill 已激活: {definition.id}",
            "skill_id": definition.id,
            "name": definition.meta.name,
            "active_skill_ids": active,
            "excerpt": definition.body[:400],
        })

    current_body_chars = 0
    for item_id in active:
        try:
            current_body_chars += len(registry.get(item_id).body)
        except (SkillNotFoundError, SkillStoreError):
            continue
    projected = current_body_chars + len(definition.body)
    if projected > MAX_ACTIVE_BODY_CHARS:
        return ToolResult.fail(
            f"激活后正文将达到 {projected} 字符，超过上限 "
            f"{MAX_ACTIVE_BODY_CHARS}。请先 unload_skill 再加载，"
            "或把该 skill 拆成更短的流程。"
        )
    try:
        session.activate_skill(definition.id)
    except SkillActivationError as exc:
        return ToolResult.fail(str(exc))
    except Exception as exc:
        return ToolResult.fail(str(exc))
    return ToolResult.success({
        "message": f"已激活 skill: {definition.id}",
        "skill_id": definition.id,
        "name": definition.meta.name,
        "allowed_tools": list(definition.meta.allowed_tools),
        "active_skill_ids": _active_ids(session),
        "excerpt": definition.body[:400],
        "note": (
            "完整正文将在下一轮临时注入，不会写入 transcript；"
            "allowed_tools 只是建议，当前会话的工具清单不会改变。"
        ),
    })


def unload_skill(
    skill_id: str,
    runtime: ToolRuntime | None = None,
    *,
    registry: SkillRegistry,
) -> ToolResult:
    del registry
    session = _session(runtime)
    try:
        removed = session.deactivate_skill(skill_id)
    except SkillStoreError as exc:
        return ToolResult.fail(str(exc))
    except Exception as exc:
        return ToolResult.fail(str(exc))
    if not removed:
        return ToolResult.fail(f"skill 未激活: {skill_id}")
    return ToolResult.success({
        "message": f"已卸载 skill: {skill_id}",
        "skill_id": skill_id,
        "active_skill_ids": _active_ids(session),
    })


def build_skill_tools(registry: SkillRegistry) -> list[Tool]:
    def bind(function):
        return lambda args, runtime: function(
            **args, runtime=runtime, registry=registry
        )

    return [
        Tool(
            name="list_skills",
            description=(
                "列出当前可用的 skill（id、名称、一句话描述）。"
                "可用 query 做关键词过滤。需要完整步骤时再 load_skill。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选关键词，匹配 id / name / description",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            call=bind(list_skills),
            is_concurrency_safe=lambda args: True,
        ),
        Tool(
            name="load_skill",
            description=(
                "激活一个 skill：下一轮会临时看到完整步骤。"
                "同时最多 3 个；用完后调用 unload_skill，避免污染后续任务。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                        "pattern": r"^[A-Za-z0-9_-]+$",
                    },
                },
                "required": ["skill_id"],
                "additionalProperties": False,
            },
            call=bind(load_skill),
        ),
        Tool(
            name="unload_skill",
            description="卸载已激活的 skill，下一轮不再注入其正文。",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                        "pattern": r"^[A-Za-z0-9_-]+$",
                    },
                },
                "required": ["skill_id"],
                "additionalProperties": False,
            },
            call=bind(unload_skill),
        ),
    ]
