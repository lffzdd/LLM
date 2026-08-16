import json
from pathlib import Path

from ...checkpoint import SessionCheckpointStore
from ...session import SessionState
from ...skills.store import write_skill


def test_checkpoint_round_trips_skill_ids_not_bodies(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_skill(
        workspace / "skills",
        "release-check",
        name="发布前检查",
        description="发布时使用",
        body="旧正文，checkpoint 不应保存它",
    )
    session = SessionState.create("goal", workspace)
    session.activate_skill("release-check")
    store = SessionCheckpointStore(tmp_path / "checkpoints")
    path = store.save(session)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session"]["active_skill_ids"] == ["release-check"]
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "旧正文" not in dumped

    restored = store.load(session.session_id)
    assert restored.get_active_skill_ids() == ["release-check"]


def test_old_checkpoint_without_skill_ids_still_loads(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = SessionState.create("goal", workspace)
    store = SessionCheckpointStore(tmp_path / "checkpoints")
    path = store.save(session)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["session"].pop("active_skill_ids", None)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    restored = store.load(session.session_id)
    assert restored.get_active_skill_ids() == []
