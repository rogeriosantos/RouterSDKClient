"""Offline tests for the router gateway: passthrough routing, classifier-driven auto
routing, decision caching, z.ai-failure fallback, header forwarding, and byte-pump
proxying (JSON + SSE). Fake asyncio servers stand in for all three backends."""
from __future__ import annotations

import asyncio
import json

import pytest

from router_warm.gateway import Gateway, last_user_signal, load_config
from router_warm.classify import Classifier, parse_reply

KEY = "test-key"


class FakeBackend:
    """Stands in for z.ai / claude-warm / codex-warm. Returns canned replies; records requests.

    If `classifier_reply` is set, requests whose body model contains "flash" (the classifier)
    get that JSON string back — letting tests script the classification decision.
    """

    def __init__(self, name: str, reply: str | None = None,
                 classifier_reply: str | None = None):
        self.name, self.classifier_reply = name, classifier_reply
        self.reply = reply or f"{name} reply"
        self.requests: list[dict] = []
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> "FakeBackend":
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return self

    @property
    def port(self) -> int:
        return self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path = lines[0].split(" ")[0], lines[0].split(" ")[1]
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or 0)
            body = await reader.readexactly(length) if length else b""
            body_json = json.loads(body) if body else {}
            self.requests.append({"method": method, "path": path, "headers": headers,
                                  "body": body_json})
            model = str(body_json.get("model") or "")
            if self.classifier_reply is not None and "flash" in model:
                payload = json.dumps({"choices": [{"message": {
                    "content": self.classifier_reply}}]}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            elif body_json.get("stream"):
                chunk = json.dumps({"choices": [{"delta": {"content": self.reply}}]})
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
                             + f"data: {chunk}\n\n".encode() + b"data: [DONE]\n\n")
            else:
                payload = json.dumps({"model": model, "choices": [{"message": {
                    "content": self.reply}}]}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


async def _http(port, method, path, body=None, headers=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    head = {"Host": "x", "Content-Length": str(len(data)), "Authorization": f"Bearer {KEY}"}
    head.update(headers or {})
    head = {k: v for k, v in head.items() if v is not None}
    writer.write((f"{method} {path} HTTP/1.1\r\n"
                  + "".join(f"{k}: {v}\r\n" for k, v in head.items()) + "\r\n").encode() + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    status = int(raw.split(b" ", 2)[1])
    return status, raw.split(b"\r\n\r\n", 1)[1].decode()


def _req(model, content="hello", stream=False):
    return {"model": model, "stream": stream,
            "messages": [{"role": "user", "content": content}]}


@pytest.fixture
async def stack(tmp_path):
    zai = await FakeBackend("zai", reply="glm reply",
                            classifier_reply='{"category": "deep_code"}').start()
    claude = await FakeBackend("claude-sdk", reply="claude reply").start()
    codex = await FakeBackend("codex-sdk", reply="codex reply").start()
    from router_warm.gateway import Backend
    backends = {
        "zai": Backend("zai", f"http://127.0.0.1:{zai.port}/v1", "zai-key", False),
        "claude-sdk": Backend("claude-sdk", f"http://127.0.0.1:{claude.port}/v1",
                              "claude-key", True),
        "codex-sdk": Backend("codex-sdk", f"http://127.0.0.1:{codex.port}/v1",
                             "codex-key", True),
    }
    gw = await Gateway(port=0, api_key=KEY, config=load_config(), backends=backends).start()
    gw.fake = {"zai": zai, "claude-sdk": claude, "codex-sdk": codex}
    yield gw
    await gw.aclose()
    for b in (zai, claude, codex):
        await b.stop()


# --- pure helpers ---------------------------------------------------------------------------

def test_parse_reply_takes_the_first_known_category():
    assert parse_reply('{"category": "quick_chat"}') == "quick_chat"
    assert parse_reply('```json\n{"category": "deep_code"}\n```') == "deep_code"
    assert parse_reply('Sure! {"category": "vision"} hope that helps') == "vision"
    assert parse_reply('{"category": "nonsense"}') is None
    assert parse_reply("") is None


def test_last_user_signal_reads_the_final_user_message():
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "text", "text": "what is"},
                                     {"type": "image_url", "image_url": {"url": "data:..."}}]},
    ]
    text, has_images = last_user_signal(msgs)
    assert text == "what is" and has_images is True


def test_route_table_passthrough_and_auto():
    gw = object.__new__(Gateway)
    gw.config = load_config()
    assert Gateway.route(gw, "glm-5.3") == ("zai", "glm-5.3")
    assert Gateway.route(gw, "claude-haiku-4-5") == ("claude-sdk", "claude-haiku-4-5")
    assert Gateway.route(gw, "gpt-6-astra") == ("codex-sdk", "gpt-6-astra")
    assert Gateway.route(gw, "router/auto") == ("", "")
    with pytest.raises(ValueError):
        Gateway.route(gw, "llama-3")


# --- HTTP -----------------------------------------------------------------------------------

async def test_health_models_auth_origin(stack):
    gw = stack
    assert (await _http(gw.port, "GET", "/health", headers={"Authorization": None}))[0] == 200
    assert (await _http(gw.port, "GET", "/v1/models",
                        headers={"Authorization": "Bearer nope"}))[0] == 401
    status, body = await _http(gw.port, "GET", "/v1/models")
    assert status == 200 and "router/auto" in body and "glm-5.3" in body
    status, _ = await _http(gw.port, "POST", "/v1/chat/completions",
                            _req("glm-5.3"), headers={"Origin": "https://evil.example"})
    assert status == 403


async def test_passthrough_hits_the_right_backend_untouched(stack):
    gw = stack
    status, body = await _http(gw.port, "POST", "/v1/chat/completions", _req("glm-5.3"))
    assert status == 200 and "glm reply" in body
    assert len(gw.fake["zai"].requests) == 1
    req = gw.fake["zai"].requests[0]
    assert req["body"]["model"] == "glm-5.3"
    assert req["path"] == "/v1/chat/completions"    # well-formed absolute path
    assert req["headers"]["authorization"] == "Bearer zai-key"
    assert "x-pi-cwd" not in req["headers"]            # z.ai is not an agent backend

    status, body = await _http(gw.port, "POST", "/v1/chat/completions", _req("gpt-6-astra"))
    assert "codex reply" in body
    assert gw.fake["codex-sdk"].requests[0]["body"]["model"] == "gpt-6-astra"


async def test_auto_classifies_then_routes_and_caches(stack):
    gw = stack
    status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                               _req("router/auto", "refactor this module for me"))
    assert status == 200 and "claude reply" in body
    zai_calls = gw.fake["zai"].requests
    classifier_calls = [r for r in zai_calls if "flash" in str(r["body"].get("model"))]
    assert len(classifier_calls) == 1                  # one classification…
    assert classifier_calls[0]["headers"]["authorization"] == "Bearer zai-key"
    claude = gw.fake["claude-sdk"].requests[-1]
    assert claude["body"]["model"] == "claude-opus-5-5"   # deep_code route
    assert claude["headers"]["authorization"] == "Bearer claude-key"

    # Same message again → cached decision, no second classifier call.
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "refactor this module for me"))
    assert len([r for r in gw.fake["zai"].requests
                if "flash" in str(r["body"].get("model"))]) == 1
    assert len(gw.fake["claude-sdk"].requests) == 2    # but the turn itself still ran


async def test_auto_forwards_pi_cwd_to_agent_backends(stack, tmp_path):
    gw = stack
    await _http(gw.port, "POST", "/v1/chat/completions", _req("router/auto"),
                headers={"X-Pi-Cwd": str(tmp_path)})
    assert gw.fake["claude-sdk"].requests[-1]["headers"].get("x-pi-cwd") == str(tmp_path)


async def test_classifier_failure_falls_back(stack):
    gw = stack
    gw.classifier.api_key = "rotten"                   # z.ai rejects → classify returns ""
    gw.classifier._cache.clear()
    # The FakeBackend ignores auth, so force failure differently: point at a dead port.
    gw.classifier.base_url = "http://127.0.0.1:1/v1"
    status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                               _req("router/auto", "totally new question"))
    assert status == 200 and "claude reply" in body    # fallback = claude-sdk/claude-opus-5-5
    assert gw.fake["claude-sdk"].requests[-1]["body"]["model"] == "claude-opus-5-5"


async def test_streaming_is_pumped_byte_for_byte(stack):
    gw = stack
    status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                               _req("glm-5.3", stream=True))
    assert status == 200
    assert "data:" in body and "[DONE]" in body and "glm reply" in body


async def test_dead_backend_is_a_502(stack):
    gw = stack
    from router_warm.gateway import Backend
    gw.backends["codex-sdk"] = Backend("codex-sdk", "http://127.0.0.1:1/v1", "k", True)
    status, body = await _http(gw.port, "POST", "/v1/chat/completions", _req("gpt-5.5"))
    assert status == 502 and "unreachable" in body


async def test_unknown_model_is_400(stack):
    assert (await _http(stack.port, "POST", "/v1/chat/completions",
                        _req("llama-3")))[0] == 400
