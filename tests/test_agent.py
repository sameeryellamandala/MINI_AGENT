from __future__ import annotations

from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import WEB_SEARCH_TOOL, SearchAgent, build_graph


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = "A checked answer with [a source](https://example.com)."
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_agent_exposes_exactly_one_browser_search_tool():
    client = FakeClient()
    graph = build_graph(checkpointer=InMemorySaver(), client=client)
    agent = SearchAgent(graph)

    agent.ask(
        thread_id="thread-one",
        question="What happened today?",
        asked_at="2026-11-10T20:30:00+05:30",
    )

    assert client.chat.completions.calls[0]["tools"] == [WEB_SEARCH_TOOL]
    assert WEB_SEARCH_TOOL == {"type": "browser_search"}


def test_thread_memory_preserves_question_and_timestamp():
    client = FakeClient()
    graph = build_graph(checkpointer=InMemorySaver(), client=client)
    agent = SearchAgent(graph)
    timestamp = "2026-11-10T20:30:00+05:30"

    agent.ask(thread_id="remember-me", question="Remember this question", asked_at=timestamp)
    history = agent.history("remember-me")

    assert history[0]["content"] == "Remember this question"
    assert history[0]["asked_at"] == timestamp
    assert history[1]["role"] == "assistant"


def test_follow_up_receives_timestamped_history():
    client = FakeClient()
    graph = build_graph(checkpointer=InMemorySaver(), client=client)
    agent = SearchAgent(graph)

    agent.ask(
        thread_id="follow-up",
        question="Who won?",
        asked_at="2026-11-10T20:30:00+05:30",
    )
    agent.ask(
        thread_id="follow-up",
        question="When did I ask that?",
        asked_at="2026-11-10T20:31:00+05:30",
    )

    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert any("2026-11-10T20:30:00+05:30" in message["content"] for message in second_call_messages)

