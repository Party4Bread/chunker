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


def test_upload_cleans_html_for_chunks_but_preserves_raw_text(tmp_path, monkeypatch):
    from app.main import create_app

    _reset_runtime(monkeypatch, tmp_path)
    client = TestClient(create_app())

    project = client.post("/api/projects", json={"name": "HTML Import"}).json()
    slug = project["slug"]
    created = client.post(
        f"/api/projects/{slug}/records/upload",
        files={
            "src_file": ("src.html", b"<p>Hello<br>world</p>", "text/html"),
            "tgt_file": ("tgt.html", b"<script>alert(1)</script><div>Hi&nbsp;&amp;&nbsp;there</div>", "text/html"),
        },
        data={"run_model": "false", "clean_html": "true"},
    ).json()

    assert created["src_text"] == "<p>Hello<br>world</p>"
    assert created["tgt_text"] == "<script>alert(1)</script><div>Hi&nbsp;&amp;&nbsp;there</div>"
    assert created["src_chunks"] == ["Hello", "world"]
    assert created["tgt_chunks"] == ["Hi & there"]
    assert created["html_cleaned_src"] is True
    assert created["html_cleaned_tgt"] is True


def test_reinfer_suffix_uses_caller_suffix_without_persisting(tmp_path, monkeypatch):
    from app.main import create_app
    from app.routers import infer

    _reset_runtime(monkeypatch, tmp_path)
    client = TestClient(create_app())
    monkeypatch.setattr(
        infer,
        "run_inference",
        lambda src, tgt: {"response": "<answer>1-1</answer>", "pairs": [(1, 1)], "parse_error": False},
    )

    project = client.post("/api/projects", json={"name": "Suffix Infer"}).json()
    slug = project["slug"]
    created = client.post(
        f"/api/projects/{slug}/records/upload",
        files={
            "src_file": ("src.txt", b"locked\nold suffix", "text/plain"),
            "tgt_file": ("tgt.txt", b"locked\nold suffix", "text/plain"),
        },
        data={"run_model": "false"},
    ).json()

    out = client.post(
        f"/api/projects/{slug}/records/{created['id']}/reinfer-suffix",
        json={"from_index": 0, "src_suffix": "new src one\n\nnew src two", "tgt_suffix": "new tgt one"},
    ).json()

    assert out["from_index"] == 0
    assert out["src_chunks"] == ["new src one", "new src two"]
    assert out["tgt_chunks"] == ["new tgt one"]
    assert out["pairs"] == [[1, 1]]

    reloaded = client.get(f"/api/projects/{slug}/records/{created['id']}").json()
    assert reloaded["src_chunks"] == created["src_chunks"]
    assert reloaded["tgt_chunks"] == created["tgt_chunks"]


def test_rechunk_below_rechunks_suffix_without_persisting(tmp_path, monkeypatch):
    from app.main import create_app
    from app.routers import infer

    _reset_runtime(monkeypatch, tmp_path)
    client = TestClient(create_app())
    monkeypatch.setattr(
        infer,
        "run_inference",
        lambda src, tgt: {"response": "<answer>1-1</answer>", "pairs": [(1, 1)], "parse_error": False},
    )

    project = client.post("/api/projects", json={"name": "Rechunk Below"}).json()
    slug = project["slug"]
    created = client.post(
        f"/api/projects/{slug}/records/upload",
        files={
            "src_file": ("src.txt", b"locked\nold suffix", "text/plain"),
            "tgt_file": ("tgt.txt", b"locked\nold suffix", "text/plain"),
        },
        data={"run_model": "false"},
    ).json()

    out = client.post(
        f"/api/projects/{slug}/records/{created['id']}/rechunk-below",
        json={
            "lock_until_pair_index": 0,
            "src_suffix_text": "New source one. New source two.",
            "tgt_suffix_text": "New target one. New target two.",
            "max_source_chars": 2000,
            "target_source_chars": 1800,
        },
    ).json()

    assert out["lock_until_pair_index"] == 0
    assert out["src_chunks"] == ["New source one. New source two."]
    assert out["tgt_chunks"] == ["New target one. New target two."]
    assert all(len(chunk) <= 2000 for chunk in out["src_chunks"])

    reloaded = client.get(f"/api/projects/{slug}/records/{created['id']}").json()
    assert reloaded["src_chunks"] == created["src_chunks"]
    assert reloaded["tgt_chunks"] == created["tgt_chunks"]
