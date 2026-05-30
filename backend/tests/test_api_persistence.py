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


def test_record_patch_persists_after_fresh_app_load(tmp_path, monkeypatch):
    from app.main import create_app

    _reset_runtime(monkeypatch, tmp_path)
    client = TestClient(create_app())

    project = client.post("/api/projects", json={"name": "Persist Test"}).json()
    slug = project["slug"]
    created = client.post(
        f"/api/projects/{slug}/records/upload",
        files={
            "src_file": ("src.txt", b"first\nsecond", "text/plain"),
            "tgt_file": ("tgt.txt", b"one\ntwo", "text/plain"),
        },
        data={"run_model": "false"},
    ).json()

    updated = client.patch(
        f"/api/projects/{slug}/records/{created['id']}",
        json={
            "title": "saved title",
            "src_chunks": ["changed source", "second source"],
            "tgt_chunks": ["changed target", "second target"],
            "gt_pairs": [[1, 1]],
            "notes": "saved notes",
        },
    )
    assert updated.status_code == 200

    _reset_runtime(monkeypatch, tmp_path)
    fresh_client = TestClient(create_app())
    reloaded = fresh_client.get(f"/api/projects/{slug}/records/{created['id']}").json()

    assert reloaded["title"] == "saved title"
    assert reloaded["src_chunks"] == ["changed source", "second source"]
    assert reloaded["tgt_chunks"] == ["changed target", "second target"]
    assert reloaded["gt_pairs"] == [[1, 1]]
    assert reloaded["notes"] == "saved notes"
