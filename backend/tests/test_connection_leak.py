"""Regression test: DB connections must be returned to the pool per-request.

The session dependencies previously did ``return next(project_session(slug))``,
which discarded the generator so its ``finally: db.close()`` ran immediately
(on refcount GC) instead of after the request. The endpoint's later query then
checked out a *fresh* connection that nothing ever closed, leaking one
connection per session per request until the pool (5 + 10 overflow) was
exhausted and every request timed out.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _reset_runtime(monkeypatch, data_dir):
    from app import config, db

    monkeypatch.setenv("CHUNKER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CHUNKER_ENABLE_SPLITTER", "false")
    config.get_settings.cache_clear()
    if db._registry_engine is not None:
        db._registry_engine.dispose()
    for engine in db._engines.values():
        engine.dispose()
    db._registry_engine = None
    db._engines.clear()


def test_connections_returned_to_pool_after_requests(tmp_path, monkeypatch):
    from app import db
    from app.main import create_app

    _reset_runtime(monkeypatch, tmp_path)
    client = TestClient(create_app())

    project = client.post("/api/projects", json={"name": "Leak Test"}).json()
    slug = project["slug"]
    created = client.post(
        f"/api/projects/{slug}/records/upload",
        files={
            "src_file": ("src.txt", b"first\nsecond", "text/plain"),
            "tgt_file": ("tgt.txt", b"one\ntwo", "text/plain"),
        },
        data={"run_model": "false"},
    ).json()

    # Hammer endpoints that depend on both the registry and project sessions.
    for _ in range(25):
        assert client.get("/api/projects").status_code == 200
        assert client.get(f"/api/projects/{slug}/records/{created['id']}").status_code == 200

    registry_checked_out = db.get_registry_engine().pool.checkedout()
    project_checked_out = db.get_project_engine(slug).pool.checkedout()

    assert registry_checked_out == 0, f"registry pool leaked {registry_checked_out} connection(s)"
    assert project_checked_out == 0, f"project pool leaked {project_checked_out} connection(s)"
