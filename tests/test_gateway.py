"""Offline tests for the router gateway: passthrough routing, classifier-driven auto
routing (Jev / TypeSafe System One), decision caching, classifier-failure fallback,
header forwarding, and byte-pump proxying (JSON + SSE). Fake asyncio servers stand in
for Jev and all three backends."""
from __future__ import annotations

import asyncio
import json

import pytest

from router_warm.gateway import Gateway, last_user_signal, load_config
from router_warm.classify import Classifier, parse_answer

KEY = "test-key"


class FakeBackend:
    """Stands in for z.ai / claude-warm / codex-warm. Returns canned replies; records requests.
    """

    def __init__(self, name: str, reply: str | None = None, delay: float = 0.0):
        self.name = name
        self.reply = reply or f"{name} reply"
        self.delay = delay
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
            if self.delay:
                await asyncio.sleep(self.delay)
            model = str(body_json.get("model") or "")
            if body_json.get("stream"):
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


class FakeJev:
    """Stands in for api.typesafe.ai POST /v1/systemone. `category`/`difficulty` script the
    choice answers; every request is recorded for assertions."""

    def __init__(self, category: str = "deep_code", difficulty: str = "standard",
                 p_category: float = 0.9, p_difficulty: float = 0.8):
        self.category = category
        self.difficulty = difficulty
        self.p_category = p_category
        self.p_difficulty = p_difficulty
        self.requests: list[dict] = []
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> "FakeJev":
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
            path = lines[0].split(" ")[1]
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            self.requests.append({"path": path, "headers": headers,
                                  "body": json.loads(body) if body else {}})
            payload = json.dumps({"model": "jev-1.13.0", "answers": {
                "category": {"type": "choice", "choice": self.category,
                             "confidence": self.p_category,
                             "probabilities": {self.category: self.p_category,
                                               "quick_chat": round(1 - self.p_category, 2)}},
                "difficulty": {"type": "choice", "choice": self.difficulty,
                               "confidence": self.p_difficulty,
                               "probabilities": {self.difficulty: self.p_difficulty,
                                                 "light": round(1 - self.p_difficulty, 2)}}}}).encode()
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
    zai = await FakeBackend("zai", reply="glm reply").start()
    claude = await FakeBackend("claude-sdk", reply="claude reply").start()
    codex = await FakeBackend("codex-sdk", reply="codex reply").start()
    jev = await FakeJev("deep_code").start()
    from router_warm.gateway import Backend
    backends = {
        "zai": Backend("zai", f"http://127.0.0.1:{zai.port}/v1", "zai-key", False),
        "claude-sdk": Backend("claude-sdk", f"http://127.0.0.1:{claude.port}/v1",
                              "claude-key", True),
        "codex-sdk": Backend("codex-sdk", f"http://127.0.0.1:{codex.port}/v1",
                             "codex-key", True),
    }
    classifier = Classifier(base_url=f"http://127.0.0.1:{jev.port}", api_key="jev-key")
    gw = await Gateway(port=0, api_key=KEY, config=load_config(), backends=backends,
                       classifier=classifier).start()
    gw.fake = {"zai": zai, "claude-sdk": claude, "codex-sdk": codex, "jev": jev}
    yield gw
    await gw.aclose()
    for b in (zai, claude, codex, jev):
        await b.stop()


# --- pure helpers ---------------------------------------------------------------------------

def test_parse_answer_takes_a_known_choice():
    def body(cat, diff="standard"):
        return {"model": "jev-1.13.0", "answers": {
            "category": {"type": "choice", "choice": cat, "confidence": 0.9},
            "difficulty": {"type": "choice", "choice": diff, "confidence": 0.8}}}

    assert parse_answer(body("quick_chat"))[:2] == ("quick_chat", "standard")
    assert parse_answer(body("design_ui", "hard"))[:2] == ("design_ui", "hard")
    assert parse_answer(body("nonsense")) is None
    assert parse_answer(body("deep_code", "wat"))[:2] == ("deep_code", "standard")  # diff default
    assert parse_answer({"answers": {"category": {"type": "choice", "choice": "vision"}}})[:2] \
        == ("vision", "standard")                      # missing difficulty → default
    assert parse_answer(body("quick_chat"))[2:] == (1.0, 1.0)   # no probabilities → confident
    assert parse_answer({}) is None
    assert parse_answer({"answers": {}}) is None
    assert parse_answer(None) is None


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
    assert len(gw.fake["zai"].requests) == 0            # classification no longer touches z.ai
    jev_calls = gw.fake["jev"].requests
    assert len(jev_calls) == 1                          # one classification…
    assert jev_calls[0]["path"] == "/v1/systemone"
    assert jev_calls[0]["headers"]["authorization"] == "Bearer jev-key"
    q = jev_calls[0]["body"]["questions"]
    assert q["category"]["type"] == "choice" and "design_ui" in q["category"]["criteria"]
    assert q["difficulty"]["type"] == "choice" and "hard" in q["difficulty"]["criteria"]
    claude = gw.fake["claude-sdk"].requests[-1]
    assert claude["body"]["model"] == "claude-sonnet-5-5"  # deep_code/standard route
    assert claude["headers"]["authorization"] == "Bearer claude-key"

    # Same message again → cached decision, no second classifier call.
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "refactor this module for me"))
    assert len(gw.fake["jev"].requests) == 1
    assert len(gw.fake["claude-sdk"].requests) == 2    # but the turn itself still ran


async def test_auto_forwards_pi_cwd_to_agent_backends(stack, tmp_path):
    gw = stack
    await _http(gw.port, "POST", "/v1/chat/completions", _req("router/auto"),
                headers={"X-Pi-Cwd": str(tmp_path)})
    assert gw.fake["claude-sdk"].requests[-1]["headers"].get("x-pi-cwd") == str(tmp_path)


async def test_difficulty_changes_the_model(stack):
    gw = stack
    gw.fake["jev"].category, gw.fake["jev"].difficulty = "deep_code", "light"
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "rename this variable everywhere"))
    assert gw.fake["claude-sdk"].requests[-1]["body"]["model"] == "claude-sonnet-5-5"

    gw.fake["jev"].category, gw.fake["jev"].difficulty = "deep_code", "hard"
    gw.classifier._cache.clear()
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "prove this concurrency refactor is race-free"))
    assert gw.fake["claude-sdk"].requests[-1]["body"]["model"] == "claude-opus-5-5"

    gw.fake["jev"].category, gw.fake["jev"].difficulty = "reasoning", "light"
    gw.classifier._cache.clear()
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "why does this loop print twice"))
    assert gw.fake["zai"].requests[-1]["body"]["model"] == "glm-5.3"


async def test_classifier_failure_falls_back(stack):
    gw = stack
    # Point at a dead port → classify returns "" → fallback route.
    gw.classifier.base_url = "http://127.0.0.1:1"
    gw.classifier._cache.clear()
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


async def test_dead_backend_is_a_502_for_explicit_models(stack):
    gw = stack
    from router_warm.gateway import Backend
    gw.backends["codex-sdk"] = Backend("codex-sdk", "http://127.0.0.1:1/v1", "k", True)
    status, body = await _http(gw.port, "POST", "/v1/chat/completions", _req("gpt-5.5"))
    assert status == 502 and "unreachable" in body   # explicit model: no silent failover


async def test_auto_fails_over_when_backend_is_down(stack):
    gw = stack
    from router_warm.gateway import Backend
    gw.backends["codex-sdk"] = Backend("codex-sdk", "http://127.0.0.1:1/v1", "k", True)
    gw.fake["jev"].category = "agent_work"           # → codex/gpt-6-luna (dead) …
    status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                               _req("router/auto", "run the test suite and fix failures"))
    assert status == 200 and "claude reply" in body   # …but the turn still succeeds
    assert gw.fake["claude-sdk"].requests[-1]["body"]["model"] == "claude-opus-5-5"  # fallback


async def test_design_ui_and_vision_routes(stack):
    gw = stack
    gw.fake["jev"].category = "design_ui"
    status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                               _req("router/auto", "improve the visual hierarchy of this page"))
    assert status == 200 and "codex reply" in body
    assert gw.fake["codex-sdk"].requests[-1]["body"]["model"] == "gpt-6-astra"

    gw.fake["jev"].category = "vision"
    gw.classifier._cache.clear()
    await _http(gw.port, "POST", "/v1/chat/completions",
                _req("router/auto", "critique this screenshot", stream=False))
    assert gw.fake["codex-sdk"].requests[-1]["body"]["model"] == "gpt-6-luna"  # vision/standard


def test_vision_and_design_difficulty_table(stack):
    gw = stack
    assert gw.target_for("vision", "light") == ("zai", "glm-5.3-flash")
    assert gw.target_for("vision", "standard") == ("codex-sdk", "gpt-6-luna")
    assert gw.target_for("vision", "hard") == ("codex-sdk", "gpt-6-astra")
    assert gw.target_for("design_ui", "light") == ("zai", "glm-5.3")
    assert gw.target_for("design_ui", "standard") == ("codex-sdk", "gpt-6-astra")
    assert gw.target_for("design_ui", "hard") == ("claude-sdk", "claude-opus-5-5")
    assert "claude-sonnet-5-5" in gw.model_ids()


async def test_media_in_history_moves_text_only_model_to_vision_route(stack):
    gw = stack
    body = _req("glm-5.3")
    body["messages"] = [
        {"role": "user", "content": "look at this"},
        {"role": "user", "content": [{"type": "text", "text": "see"},
                                     {"type": "image_url", "image_url": {"url": "data:x"}}]},
    ]
    status, _ = await _http(gw.port, "POST", "/v1/chat/completions", body)
    assert status == 200
    sent = gw.fake["codex-sdk"].requests[-1]["body"]      # bumped to vision/standard
    assert sent["model"] == "gpt-6-luna"
    assert sent["messages"][1]["content"][1]["type"] == "image_url"   # image kept


async def test_unknown_model_is_400(stack):
    assert (await _http(stack.port, "POST", "/v1/chat/completions",
                        _req("llama-3")))[0] == 400


def test_text_only_messages_strips_images_from_history():
    from router_warm.gateway import text_only_messages
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "look"},
                                     {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image_url"}]},
        {"role": "user", "content": "hello"},
    ]
    out = text_only_messages(msgs)
    assert out[0]["content"] == "look\n[image_url omitted]"
    assert out[1] == {"role": "tool", "tool_call_id": "c1", "content": "[image_url omitted]"}
    assert out[2] == msgs[2]
    assert msgs[0]["content"][1]["type"] == "image_url"   # input untouched


async def test_old_media_is_stripped_and_stays_on_text_model(stack):
    gw = stack
    body = _req("glm-5.3")
    body["messages"] = [
        {"role": "user", "content": [{"type": "text", "text": "see"},
                                     {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now just chat"},
    ]
    status, _ = await _http(gw.port, "POST", "/v1/chat/completions", body)
    assert status == 200
    sent = gw.fake["zai"].requests[-1]["body"]
    assert sent["model"] == "glm-5.3"
    assert sent["messages"][0]["content"] == "see\n[image_url omitted]"


# --- /v1/status ------------------------------------------------------------------------------

async def test_status_lists_recent_and_counters(stack):
    gw = stack
    await _http(gw.port, "POST", "/v1/chat/completions", _req("glm-5.3"))
    status, body = await _http(gw.port, "GET", "/v1/status")
    assert status == 200
    data = json.loads(body)
    assert data["counters"]["requests"] >= 1
    assert data["inflight"] == []                      # request already finished
    last = data["recent"][-1]
    assert last["model"] == "glm-5.3" and last["phase"] == "done"
    assert last["bytes"] > 0 and last["total_s"] is not None


async def test_status_shows_inflight_waiting(stack, tmp_path):
    gw = stack
    import asyncio as aio
    from router_warm.gateway import Backend
    slow = await FakeBackend("claude-sdk", reply="slow reply", delay=1.0).start()
    gw.backends["claude-sdk"] = Backend("claude-sdk", f"http://127.0.0.1:{slow.port}/v1",
                                        "claude-key", True)
    gw.fake["jev"].category = "deep_code"             # → claude (the slow fake)
    task = aio.create_task(_http(gw.port, "POST", "/v1/chat/completions",
                                 _req("router/auto", "something brand new to classify")))
    await aio.sleep(0.4)                              # mid-request now
    status, body = await _http(gw.port, "GET", "/v1/status")
    data = json.loads(body)
    assert status == 200 and len(data["inflight"]) == 1
    fl = data["inflight"][0]
    assert fl["phase"] in ("classify", "routed", "waiting")
    assert fl["backend"] == "claude-sdk" and fl["category"] == "deep_code"
    assert fl["elapsed_s"] >= 0.3
    await task                                        # let it finish cleanly
    await slow.stop()


# --- calibration: hesitant answers must not buy the strongest model --------------------------

async def test_hesitant_deep_code_downgrades_to_reasoning(stack, tmp_path):
    """Jev says deep_code but only at p=0.4 → route the cheaper reasoning tier, not opus."""
    gw = stack
    gw.fake["jev"].category = "deep_code"
    gw.fake["jev"].p_category = 0.4
    _status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                                _req("router/auto", "build a cache layer"))
    data = json.loads(body)
    route = gw.backends  # target_for is what decides; assert via the routed backend model
    name, model = gw.target_for("reasoning", "standard")
    assert data["model"] == model, f"expected reasoning-tier {model}, got {data['model']}"


async def test_hesitant_hard_downgrades_to_standard(stack, tmp_path):
    """Jev says hard at p=0.5 (< 0.6 gate) → standard difficulty model, never opus."""
    gw = stack
    gw.fake["jev"].category = "deep_code"
    gw.fake["jev"].difficulty = "hard"
    gw.fake["jev"].p_category = 0.9          # category is confident
    gw.fake["jev"].p_difficulty = 0.5        # …but the escalation is not
    _status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                                _req("router/auto", "fix this race"))
    data = json.loads(body)
    _, model = gw.target_for("deep_code", "standard")
    assert data["model"] == model, f"expected standard-tier {model}, got {data['model']}"


async def test_confident_hard_still_goes_to_opus(stack, tmp_path):
    """The gate must not soften genuinely confident escalations."""
    gw = stack
    gw.fake["jev"].category = "deep_code"
    gw.fake["jev"].difficulty = "hard"
    gw.fake["jev"].p_category = 0.95
    gw.fake["jev"].p_difficulty = 0.9
    _status, body = await _http(gw.port, "POST", "/v1/chat/completions",
                                _req("router/auto", "redesign the auth architecture"))
    data = json.loads(body)
    _, model = gw.target_for("deep_code", "hard")
    assert data["model"] == model


def test_calibrate_unit():
    from router_warm.classify import calibrate
    # confident escalations pass through untouched
    assert calibrate("deep_code", "hard", 0.9, 0.8) == ("deep_code", "hard", "")
    # hesitant difficulty drops to standard
    assert calibrate("deep_code", "hard", 0.9, 0.45)[1] == "standard"
    # hesitant deep_code drops to reasoning
    assert calibrate("deep_code", "hard", 0.3, 0.9)[:2] == ("reasoning", "hard")
    # other categories never touched
    assert calibrate("quick_chat", "light", 0.3, 0.3)[:2] == ("quick_chat", "light")


def test_parse_answer_missing_probabilities_defaults_confident():
    body = {"answers": {"category": {"choice": "quick_chat"},
                        "difficulty": {"choice": "light"}}}
    parsed = parse_answer(body)
    assert parsed == ("quick_chat", "light", 1.0, 1.0)
