"""A local OpenAI-compatible embeddings server, for code embeddings that are only observed (#56).

With ``embedding_model`` set, ShinkaEvolve embeds every program's code. The vectors go into the
database and feed its web viewer (the Embed and Clusters tabs). In the pinned Shinka they change
the search only through the novelty judge, which exists only with ``novelty_llm_models``, and
through the crowding archive. MEvolve sets neither, so the embeddings are observed and never used
(Han, 2026-11-04; study 40 r1: an embedding novelty filter would flag most proposals of this program
family).

The model is Alibaba-NLP/gte-modernbert-base at a pinned revision, on CPU: the CLS vector of the last
layer, L2-normalised, as in study 40's post hoc analysis. It runs in its own process, from a Python
with torch and transformers (the search env has neither), and serves ``POST /v1/embeddings`` in
OpenAI's format on 127.0.0.1. ``run.main`` starts it with

    python -m mevolve.search.embed_server

and it prints ``port <n>`` to stdout once the model is loaded (about a minute). The model must be in
the Hugging Face cache the job names in ``HF_HOME``: ``run.main`` sets ``HF_HUB_OFFLINE``, so a run
never downloads it.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL, REVISION = "Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
HOST = "127.0.0.1"


def make_server(embed, model: str, port: int = 0) -> HTTPServer:
    """An HTTP server on 127.0.0.1 answering OpenAI embedding requests for ``model`` with
    ``embed(texts) -> (vectors, n_tokens)``; one request at a time."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/embeddings":
                return self.send(404, {"error": {"message": f"no route {self.path}"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                texts = [body["input"]] if isinstance(body["input"], str) else list(body["input"])
                if body.get("model") != model or not texts or not all(isinstance(t, str) for t in texts):
                    raise ValueError(f"this server embeds strings with {model}")
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {"error": {"message": str(e)}})
            vectors, n_tokens = embed(texts)
            self.send(200, {"object": "list", "model": model,
                            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
                            "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens}})

        def send(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):       # one line per proposal is noise in the run's log
            pass

    return HTTPServer((HOST, port), Handler)


def load_embed(threads: int):
    """``embed(texts) -> (vectors, n_tokens)`` with the pinned model on CPU."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    net = AutoModel.from_pretrained(MODEL, revision=REVISION, attn_implementation="sdpa", reference_compile=False,
                                    use_safetensors=True, torch_dtype=torch.float32).eval()
    if net.config._commit_hash != REVISION:
        raise SystemExit(f"{MODEL} loaded at {net.config._commit_hash}, not {REVISION}")

    def embed(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.inference_mode():
            out = net(**batch)
        return F.normalize(out.last_hidden_state[:, 0], p=2, dim=1).tolist(), int(batch["attention_mask"].sum())
    return embed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for the model")
    args = parser.parse_args(argv)
    server = make_server(load_embed(args.threads), MODEL)
    print(f"port {server.server_address[1]}", flush=True)
    sys.stdout = sys.stderr         # nobody reads the pipe after the port; a full pipe would block the server
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
