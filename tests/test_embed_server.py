"""Code embeddings are served locally and only observed (#56)."""
import asyncio
import json
import os
import sqlite3
import stat
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip("shinka")

from mevolve.search import embed_server, run, seeds  # noqa: E402

MODEL = "test/model"


def fake_embed(texts):
    return [[float(len(t)), 1.0] for t in texts], sum(len(t) for t in texts)


@pytest.fixture
def server():
    srv = embed_server.make_server(fake_embed, MODEL)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"local/{MODEL}@http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_shinkas_clients_read_the_servers_embeddings(server):
    from shinka.embed import AsyncEmbeddingClient, EmbeddingClient

    assert EmbeddingClient(server).get_embedding("abc") == ([3.0, 1.0], 0.0)       # a local model costs nothing
    assert EmbeddingClient(server).get_embedding(["ab", "abcd"])[0] == [[2.0, 1.0], [4.0, 1.0]]
    vector, _ = asyncio.run(AsyncEmbeddingClient(server).embed_async("abcde"))      # what the runner calls
    assert vector == [5.0, 1.0]


def test_a_request_for_another_model_is_refused(server):
    url = server.split("@")[1] + "/embeddings"
    request = urllib.request.Request(url, data=json.dumps({"model": "other", "input": "x"}).encode(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request)
    assert e.value.code == 400


def test_seeds_get_their_embedding(tmp_path, server):
    from shinka.database import DatabaseConfig

    metrics = {"combined_score": 70.0, "public": {}, "private": {}, "text_feedback": ""}
    seeds.place(tmp_path, DatabaseConfig(num_islands=2), [{"name": "s", "code": "x = 1\n", "island": 1, "metrics": metrics}],
                server)
    row = sqlite3.connect(tmp_path / "programs.sqlite").execute("SELECT embedding FROM programs").fetchone()
    assert json.loads(row[0]) == [6.0, 1.0]


def test_embeddings_never_steer_the_search(tmp_path, server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    config = {**run.DEFAULTS, "results_dir": str(tmp_path), "max_usd": 1.0, "freeze": "/f", "cohorts": "/c",
              "generation": "g", "validation": "v"}
    runner, db = run.build_runner(config, "TASK", server)
    assert runner.evo_config.embedding_model == server and runner.embedding_client is not None
    assert runner.novelty_judge is None and not runner.evo_config.novelty_llm_models    # nothing is ever rejected
    assert db.archive_selection_strategy == "fitness"                                   # not the crowding archive
    runner, _ = run.build_runner(config, "TASK")                                        # off by default
    assert runner.embedding_client is None and run.DEFAULTS["embedding_python"] is None


def script(tmp_path, body):
    path = tmp_path / "python"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_the_server_process_starts_and_stops_with_the_run(tmp_path):
    with run.embedding_server(None) as model:
        assert model is None
    fake = script(tmp_path, f'echo $$ > {tmp_path}/pid\necho "$HF_HUB_OFFLINE" > {tmp_path}/offline\necho "port 4321"\n'
                            "exec sleep 600\n")
    with run.embedding_server(fake) as model:
        assert model == f"local/{embed_server.MODEL}@http://127.0.0.1:4321/v1"
        pid = int((tmp_path / "pid").read_text())
    assert (tmp_path / "offline").read_text().strip() == "1"           # a run never downloads the model
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_server_that_dies_stops_the_run(tmp_path):
    with pytest.raises(RuntimeError, match="did not start"):
        with run.embedding_server(script(tmp_path, "exit 3\n")):
            pass


def test_a_seed_without_its_embedding_stops_the_run(tmp_path):
    from shinka.database import DatabaseConfig

    srv = embed_server.make_server(lambda texts: ([[] for _ in texts], 0), MODEL)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    metrics = {"combined_score": 70.0, "public": {}, "private": {}, "text_feedback": ""}
    try:
        with pytest.raises(RuntimeError, match="returned nothing for seed s"):
            seeds.place(tmp_path, DatabaseConfig(num_islands=2), [{"name": "s", "code": "x = 1\n", "island": 1,
                        "metrics": metrics}], f"local/{MODEL}@http://127.0.0.1:{srv.server_address[1]}/v1")
    finally:
        srv.shutdown()
        srv.server_close()
