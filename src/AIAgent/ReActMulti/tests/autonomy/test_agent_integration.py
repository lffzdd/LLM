import json
import queue

from ...agent import Agent
from ...autonomy import AutonomyScheduler, AutonomyStore, TriggerSpec
from ...events import ContentDone
from ...main import _execute_durable_run, _resume_durable_run
from ...renderer import SilentRenderer
from ...session import SessionState


def _final(answer):
    return json.dumps({"tool_calls": [], "final_answer": answer})


class ScriptLLM:
    context_limit = 128_000

    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, messages):
        yield ContentDone(_final(self.answers.pop(0)))


def _runtime(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = AutonomyStore(
        tmp_path / "tasks.sqlite3",
        session_id="session",
        workspace_dir=workspace,
    )
    session = SessionState.create("interactive", workspace)
    session.session_id = "session"
    session.durable_task_store = store
    scheduler = AutonomyScheduler(store, queue.Queue(), poll_interval=1)
    return workspace, store, session, scheduler


def test_autonomous_event_starts_fresh_turn_and_commits_run(tmp_path):
    workspace, store, session, scheduler = _runtime(tmp_path)
    automation = store.create_automation(
        name="daily review",
        prompt="review the repository",
        trigger=TriggerSpec(type="once", run_at=0),
        now=0,
    )
    run_id = store.materialize_due(now=0)[0]
    claimed = store.claim_next_run(now=0)
    assert claimed is not None and claimed.id == run_id
    agent = Agent(
        ScriptLLM(["interactive result", "autonomous result"]),
        [],
        session,
        SilentRenderer(),
    )
    assert agent.run("interactive turn") == "interactive result"
    previous_root_turn_id = session.agent_root_turn_id
    session.plan_manager.create_plan("old", ["old step"])

    _execute_durable_run(agent, scheduler, run_id)

    run = store.get_run(run_id)
    assert run.status == "completed"
    assert run.result == "autonomous result"
    assert run.root_turn_id == session.agent_root_turn_id
    assert run.root_turn_id != previous_root_turn_id
    assert session.user_goal == "review the repository"
    assert session.active_durable_run_id is None
    assert session.plan_manager.snapshot()["steps"] == []
    assert run.automation_id == automation.id
    store.close()


def test_autonomous_run_observes_durable_cancellation_before_llm_call(tmp_path):
    workspace, store, session, scheduler = _runtime(tmp_path)
    store.create_automation(
        name="cancel me",
        prompt="should not execute",
        trigger=TriggerSpec(type="once", run_at=0),
        now=0,
    )
    run_id = store.materialize_due(now=0)[0]
    store.claim_next_run(now=0)
    store.cancel_run(run_id, "external cancellation")
    agent = Agent(
        ScriptLLM([]),
        [],
        session,
        SilentRenderer(),
    )

    _execute_durable_run(agent, scheduler, run_id)

    run = store.get_run(run_id)
    assert run.status == "cancelled"
    assert run.cancel_reason == "external cancellation"
    assert session.status == "running"
    store.close()


def test_checkpoint_owned_autonomous_turn_continues_same_run(tmp_path):
    workspace, store, session, scheduler = _runtime(tmp_path)
    store.create_automation(
        name="resume",
        prompt="continue after restart",
        trigger=TriggerSpec(type="once", run_at=0),
        now=0,
    )
    run_id = store.materialize_due(now=0)[0]
    dispatched = store.claim_next_run(now=0)
    assert dispatched is not None
    store.start_run(run_id, now=0)
    agent = Agent(
        ScriptLLM(["resumed result"]),
        [],
        session,
        SilentRenderer(),
    )
    session.active_durable_run_id = run_id
    session.begin_user_turn("continue after restart")
    session.append_message({
        "role": "user",
        "content": json.dumps({"runtime_event": scheduler.runtime_event(run_id)}),
    })
    store.set_run_root_turn(run_id, session.agent_root_turn_id)

    _resume_durable_run(agent, scheduler, run_id)

    run = store.get_run(run_id)
    assert run.status == "completed"
    assert run.result == "resumed result"
    assert run.attempt == 1
    assert session.active_durable_run_id is None
    store.close()
