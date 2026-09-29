"""router-warm gateway: an OpenAI-compatible thin router in front of the warm stack.

    python -m router_warm.gateway --port 8795

Endpoints:
  GET  /health               liveness (unauthenticated)
  GET  /v1/models            every model this router can serve
  POST /v1/chat/completions  routed and proxied byte-for-byte

Routing:
  * Explicit models pass straight through: `glm-*` → the z.ai API, `claude-*` → the
    claude-warm gateway (:8793), `gpt-*` → the codex-warm gateway (:8794). No
    classification, no delay.
  * `router/auto` asks z.ai first — one tiny GLM call classifies the last user message
    into quick_chat / quick_code / reasoning / deep_code / agent_work / vision — and a
    category→(backend, model) table (editable in ~/.config/router-warm/config.json)
    picks the target. Decisions are cached per message text; if z.ai is unreachable the
    fallback route is used so the router never becomes the outage.

Proxying is a byte pump: the backend's status line, headers, SSE stream or JSON body go
to the client untouched. `X-Pi-Cwd` is forwarded to the two agent gateways so Claude Code
/ Codex keep working in pi's directory. A client disconnect tears down the backend
connection too.

Security: binds 127.0.0.1, requires the bearer key in `~/.config/router-warm/gateway.key`
(created 0600 on first start), and rejects any request carrying an `Origin` header.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classify import Classifier, log_decision

log = logging.getLogger("router_warm.gateway")

DEFAULT_PORT = 8795
CONFIG_DIR = Path.home() / ".config" / "router-warm"
KEY_FILE = CONFIG_DIR / "gateway.key"
ZAI_KEY_FILE = CONFIG_DIR / "zai.key"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict = {
    "classifier": {"model": "glm-5.3-flash", "timeout_s": 8},
    "fallback": {"backend": "claude-sdk", "model": "claude-opus-5-5"},
    "routes": {
        "quick_chat": {"backend": "zai", "model": "glm-5.3"},
        "quick_code": {"backend": "zai", "model": "glm-5.3"},
        "reasoning": {"backend": "zai", "model": "glm-5.3"},
        "deep_code": {"backend": "claude-sdk", "model": "claude-opus-5-5"},
        "agent_work": {"backend": "claude-sdk", "model": "claude-opus-5-5"},
        "vision": {"backend": "claude-sdk", "model": "claude-opus-5-5"},
    },
}

AUTO_MODEL = "router/auto"
PASSTHROUGH = (  # prefix → backend name
    ("glm-", "zai"),
    ("claude-", "claude-sdk"),
    ("gpt-", "codex-sdk"),
)

ZAI_BASE = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
CLAUDE_BASE = os.environ.get("CLAUDE_WARM_URL", "http://127.0.0.1:8793/v1")
CODEX_BASE = os.environ.get("CODEX_WARM_URL", "http://127.0.0.1:8794/v1")

_REASON = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
           404: "Not Found", 500: "Internal Server Error", 502: "Bad Gateway"}


# ------------------------------------------------------------------------------ helpers
def load_or_create_key(path: Path = KEY_FILE) -> str:
    try:
        key = path.read_text().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key + "\n")
    return key


def read_key_file(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


def load_config(path: Path = CONFIG_FILE) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))          # deep copy
    try:
        user = json.loads(path.read_text())
        for section in ("classifier", "fallback"):
            if isinstance(user.get(section), dict):
                cfg[section].update(user[section])
        if isinstance(user.get("routes"), dict):
            cfg["routes"].update(user["routes"])
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("ignoring bad config %s: %s", path, exc)
    return cfg


@dataclass
class Backend:
    name: str
    base_url: str          # …/v1 for the gateways, …/paas/v4 for z.ai
    api_key: str
    forwards_cwd: bool     # agent gateways take the working directory

    def target(self, path: str = "/chat/completions") -> tuple[str, int, str, bool]:
        u = self.base_url.split("://", 1)[1]
        host, _, _path = u.partition("/")
        tls = self.base_url.startswith("https://")
        port = 443 if tls else 80
        if ":" in host and not host.startswith("["):
            host, port_s = host.rsplit(":", 1)
            port = int(port_s)
        return host, port, "/" + _path + path, tls


def _error(message: str, err_type: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": err_type, "param": None, "code": None}}


def last_user_signal(messages) -> tuple[str, bool]:
    """(text of the last user message, whether it carries an image) — the classifier input."""
    text, has_images = "", False
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
            has_images = False
        elif isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            text = "\n".join(str(t) for t in parts if t)
            has_images = any(isinstance(p, dict) and p.get("type") == "image_url"
                             for p in content)
    return text, has_images


async def _read_request(reader: asyncio.StreamReader):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    try:
        method, path, _ = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ", 2)
    except ValueError:
        return None
    headers: dict[str, str] = {}
    for line in head.decode("latin-1").split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    body = b""
    length = int(headers.get("content-length", "0") or 0)
    if length:
        try:
            body = await reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionError):
            return None
    return method, path.split("?", 1)[0], headers, body


# ------------------------------------------------------------------------------ gateway
class Gateway:
    def __init__(self, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT, api_key: str,
                 config: dict | None = None, backends: dict[str, Backend] | None = None,
                 classifier: Classifier | None = None):
        self.host, self.port, self.api_key = host, port, api_key
        self.config = config or load_config()
        self.backends = backends or default_backends()
        self.classifier = classifier or Classifier(
            base_url=self.backends["zai"].base_url,
            api_key=self.backends["zai"].api_key,
            model=self.config["classifier"]["model"],
            timeout_s=float(self.config["classifier"].get("timeout_s", 8)))
        self._ssl = ssl.create_default_context()
        self._server: asyncio.AbstractServer | None = None

    def route(self, model: str) -> tuple[str, str]:
        """→ (backend_name, backend_model) for a requested model id."""
        for prefix, backend in PASSTHROUGH:
            if model.startswith(prefix):
                return backend, model
        if model == AUTO_MODEL:
            return "", ""                    # decided per request by the classifier
        raise ValueError(f"unknown model {model!r}")

    def target_for(self, category: str) -> tuple[str, str]:
        route = self.config["routes"].get(category) or self.config["fallback"]
        return route["backend"], route["model"]

    async def start(self) -> "Gateway":
        self._server = await asyncio.start_server(self._on_client, self.host, self.port,
                                                  limit=1 << 20)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def aclose(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # --------------------------------------------------------------------------- plumbing
    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = await _read_request(reader)
            if req is None:
                return
            method, path, headers, body = req
            if path == "/health":
                self._send_json(writer, 200, {"status": "ok"})
            elif "origin" in headers:
                self._send_json(writer, 403, _error("browser requests are not accepted"))
            elif not secrets.compare_digest(self._bearer(headers), self.api_key):
                self._send_json(writer, 401, _error("missing or invalid API key"))
            elif path == "/v1/models" and method == "GET":
                self._send_json(writer, 200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "router"}
                    for m in self.model_ids()]})
            elif path == "/v1/chat/completions" and method == "POST":
                await self._chat(body, headers, reader, writer)
            else:
                self._send_json(writer, 404, _error(f"unknown path {method} {path}"))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as exc:  # noqa: BLE001 — never leak a traceback to the client
            log.exception("request failed")
            try:
                self._send_json(writer, 500, _error(f"gateway error: {exc}", "server_error"))
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    def model_ids(self) -> list[str]:
        ids = [AUTO_MODEL]
        ids += [f"glm-{m}" for m in ("5.3", "5.3-flash", "5.3-highspeed", "5.2", "4.7")]
        ids += ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5", "claude-haiku-4-5"]
        ids += ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
                "gpt-5.6-luna", "gpt-5.5"]
        return ids

    @staticmethod
    def _bearer(headers: dict[str, str]) -> str:
        auth = headers.get("authorization", "")
        return auth[7:].strip() if auth.lower().startswith("bearer ") else ""

    @staticmethod
    def _send_json(writer: asyncio.StreamWriter, status: int, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        writer.write((f"HTTP/1.1 {status} {_REASON.get(status, 'OK')}\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                      f"Connection: close\r\n\r\n").encode() + payload)

    # ------------------------------------------------------------------------ one request
    async def _chat(self, body: bytes, headers: dict[str, str],
                    reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(req, dict):
                raise ValueError("request body must be a JSON object")
            requested = str(req.get("model") or "")
            backend_name, backend_model = self.route(requested)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(writer, 400, _error(f"bad request: {exc}"))
            return

        if backend_name == "":                     # router/auto → classify
            text, has_images = last_user_signal(req.get("messages"))
            t0 = time.monotonic()
            category, cached = await self.classifier.classify(text, has_images)
            backend_name, backend_model = self.target_for(category)
            log_decision(log, category, cached, f"{backend_name}/{backend_model}",
                         int((time.monotonic() - t0) * 1000))

        backend = self.backends.get(backend_name)
        if backend is None:
            self._send_json(writer, 500, _error(f"unknown backend {backend_name!r}",
                                                "server_error"))
            return
        req["model"] = backend_model
        try:
            out_body = json.dumps(req).encode()
        except (TypeError, ValueError) as exc:
            self._send_json(writer, 400, _error(f"unserializable request: {exc}"))
            return

        extra = {}
        if backend.forwards_cwd:
            cwd = headers.get("x-pi-cwd")
            if cwd:
                extra["X-Pi-Cwd"] = cwd

        await self._proxy(backend, out_body, extra, reader, writer)

    async def _proxy(self, backend: Backend, body: bytes, extra: dict[str, str],
                     reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        host, port, path, tls = backend.target()
        try:
            kwargs: dict[str, Any] = {"ssl": self._ssl} if tls else {}
            b_reader, b_writer = await asyncio.open_connection(host, port, **kwargs)
        except (OSError, ConnectionError) as exc:
            log.warning("backend %s unreachable: %s", backend.name, exc)
            self._send_json(writer, 502, _error(f"backend {backend.name} unreachable: {exc}",
                                                "server_error"))
            return

        head = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
                f"Authorization: Bearer {backend.api_key}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                + "".join(f"{k}: {v}\r\n" for k, v in extra.items())
                + "\r\n").encode()
        b_writer.write(head + body)
        await b_writer.drain()

        client_gone = asyncio.Event()

        async def watch_client() -> None:
            try:
                await reader.read()               # b"" when pi closes the socket
            except Exception:  # noqa: BLE001
                pass
            client_gone.set()

        watcher = asyncio.create_task(watch_client())
        try:
            while True:
                chunk = await b_reader.read(65536)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
                if client_gone.is_set():
                    b_writer.close()              # stop the upstream turn early
                    break
        except (ConnectionError, RuntimeError):
            pass                                  # client vanished mid-stream
        finally:
            watcher.cancel()
            try:
                b_writer.close()
            except Exception:  # noqa: BLE001
                pass


def default_backends() -> dict[str, Backend]:
    zai_key = os.environ.get("ZAI_API_KEY") or read_key_file(ZAI_KEY_FILE)
    return {
        "zai": Backend("zai", ZAI_BASE, zai_key, forwards_cwd=False),
        "claude-sdk": Backend("claude-sdk", CLAUDE_BASE,
                              read_key_file(Path.home() / ".config" / "claude-warm"
                                            / "gateway.key"), forwards_cwd=True),
        "codex-sdk": Backend("codex-sdk", CODEX_BASE,
                             read_key_file(Path.home() / ".config" / "codex-warm"
                                           / "gateway.key"), forwards_cwd=True),
    }


async def serve(host: str, port: int, key_file: Path) -> None:
    gw = await Gateway(host=host, port=port, api_key=load_or_create_key(key_file)).start()
    routes = ", ".join(f"{k}→{v['backend']}/{v['model']}"
                       for k, v in gw.config["routes"].items())
    log.info("router-warm gateway on http://%s:%s/v1 (classifier %s; fallback %s/%s)",
             host, gw.port, gw.classifier.model, gw.config["fallback"]["backend"],
             gw.config["fallback"]["model"])
    log.info("routes: %s", routes)
    try:
        await asyncio.Event().wait()
    finally:
        await gw.aclose()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="router-warm-gateway", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--key-file", type=Path, default=KEY_FILE)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(serve(args.host, args.port, args.key_file))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
