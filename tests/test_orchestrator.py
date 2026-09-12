from psiduck.agents import DELEGATE_SPECS
from psiduck.orchestrator import LatentDelegationOrchestrator
from tests.fake_backbone import FakeBackbone


def make_responder():
    """Delegates propose mostly 116; verifier disagrees; aggregator boxes 116."""

    def responder(system: str, user: str) -> str:
        s = system.lower()
        if "controller" in s:
            return "- step 1\n- step 2\n- step 3"
        if "aggregator" in s:
            return r"The consensus is clear. \boxed{116}"
        if "algebra" in s:
            return r"Algebra gives \boxed{116}."
        if "combinatorics" in s:
            return r"Counting gives \boxed{116}."
        if "number theory" in s:
            return r"Mod arithmetic gives \boxed{116}."
        if "verifier" in s:
            return r"I get a different value \boxed{117}."
        return r"\boxed{0}"

    return responder


def test_all_agents_share_one_backbone():
    model = FakeBackbone(make_responder())
    orch = LatentDelegationOrchestrator(model)
    agents = [orch.controller, *orch.delegates, orch.aggregator]
    ids = {id(a.model) for a in agents}
    assert len(ids) == 1, "all agents must share a single backbone instance"
    assert orch.agent_count == 6  # controller + 4 delegates + aggregator
    assert len(orch.delegates) == len(DELEGATE_SPECS) == 4


def test_solve_uses_aggregator_answer():
    model = FakeBackbone(make_responder())
    orch = LatentDelegationOrchestrator(model)
    result = orch.solve(
        "Sample problem?",
        plan_max_new_tokens=8,
        delegate_max_new_tokens=8,
        aggregator_max_new_tokens=8,
    )
    assert result.final_answer == 116
    assert len(result.candidates) == 4
    assert result.votes.get(116) == 3
    assert result.votes.get(117) == 1
    assert result.shared_backbone_id == id(model)


def test_falls_back_to_majority_when_aggregator_unparseable():
    def responder(system: str, user: str) -> str:
        s = system.lower()
        if "controller" in s:
            return "plan"
        if "aggregator" in s:
            return "I am not sure, no number here."
        if "verifier" in s:
            return r"\boxed{200}"
        return r"\boxed{42}"  # the three non-verifier delegates

    model = FakeBackbone(responder)
    orch = LatentDelegationOrchestrator(model)
    result = orch.solve("p?")
    # aggregator gave nothing parseable -> majority of delegates (42 x3)
    assert result.final_answer == 42


def test_delegates_receive_plan_in_prompt():
    model = FakeBackbone(make_responder())
    orch = LatentDelegationOrchestrator(model)
    orch.solve("Special problem statement")
    delegate_calls = [c for c in model.calls if "specialty" in c["user"]]
    assert len(delegate_calls) == 4
    for call in delegate_calls:
        assert "Special problem statement" in call["user"]
        assert "Team plan" in call["user"]
