from feedback_loop import api_v2
from feedback_loop.api import EvalResult
from feedback_loop.v2_state import TaskState


def test_run_iteration_records_successful_v2_attempt(monkeypatch):
    # Arrange
    state = TaskState(
        task_id="16",
        entry_point="Bell_State",
        category="state",
        model="local/test",
        prompt="Build a Bell state.",
        signature_prefill="def Bell_State():",
    )
    monkeypatch.setattr(
        api_v2,
        "send_requests_in_parallel",
        lambda requests: [{"choices": [{"message": {"content": "ok"}}]}],
    )
    monkeypatch.setattr(api_v2, "extract_code", lambda raw, state: ({"code": "code"}, "code"))
    monkeypatch.setattr(
        api_v2,
        "evaluate_state",
        lambda **kwargs: EvalResult(compiled=True, ran=True, kl_div_bool=True),
    )

    # Act
    records = api_v2.run_iteration(
        [state],
        framework="cirq",
        benchmark_version="v2",
        evaluator=object(),
        feedback_num=1,
    )

    # Assert
    assert state.done is True
    assert state.attempts_used == 1
    assert records[0]["framework"] == "cirq"
    assert records[0]["evaluation"]["benchmark_version"] == "v2"
    assert records[0]["feedback_sent_to_model"] is None


def test_run_iteration_adds_feedback_after_failed_v2_attempt(monkeypatch):
    # Arrange
    state = TaskState(
        task_id="16",
        entry_point="Bell_State",
        category="state",
        model="local/test",
        prompt="Build a Bell state.",
        signature_prefill="def Bell_State():",
    )
    monkeypatch.setattr(
        api_v2,
        "send_requests_in_parallel",
        lambda requests: [{"choices": [{"message": {"content": "bad"}}]}],
    )
    monkeypatch.setattr(api_v2, "extract_code", lambda raw, state: (None, "bad code"))
    monkeypatch.setattr(
        api_v2,
        "evaluate_state",
        lambda **kwargs: EvalResult(compiled=True, ran=True, kl_div_bool=False, output=[1.0, 0.0]),
    )

    # Act
    records = api_v2.run_iteration(
        [state],
        framework="pennylane",
        benchmark_version="v2",
        evaluator=object(),
        feedback_num=1,
    )

    # Assert
    assert state.done is False
    assert state.history[0]["assistant_code"] == "bad code"
    assert "does NOT match" in state.history[0]["feedback_to_model"]
    assert records[0]["framework"] == "pennylane"
    assert records[0]["feedback_sent_to_model"] == state.last_feedback
