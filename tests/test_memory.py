from agent.memory import ThreadRegistry


def test_registry_lists_latest_thread_first(tmp_path):
    registry = ThreadRegistry(tmp_path / "memory.db")
    registry.touch(thread_id="old", title="Old question", timestamp="2026-01-01T00:00:00+00:00")
    registry.touch(thread_id="new", title="New question", timestamp="2026-01-02T00:00:00+00:00")

    threads = registry.list_threads()

    assert [thread["thread_id"] for thread in threads] == ["new", "old"]

