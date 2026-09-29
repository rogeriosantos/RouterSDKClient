"""The z.ai classifier: one small GLM call turns a request into a routing category.

Strict-JSON, low temperature, defensively parsed (models sometimes wrap the object in
fences or prose). Decisions are cached by (text, has_images) so a multi-turn conversation
— where the client resends the transcript — classifies once, not per turn.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import ssl
import time
from collections import OrderedDict
from typing import Any

CATEGORIES = ("quick_chat", "quick_code", "reasoning", "deep_code", "agent_work", "vision",
             "design_ui")

_SYSTEM = (
    "You are a routing classifier for a coding assistant. Read the user's request and reply "
    "with ONLY minified JSON: {\"category\": \"<one>\"}. Pick exactly one category:\n"
    "- quick_chat: greetings, simple Q&A, facts, small talk\n"
    "- quick_code: short snippets, syntax questions, tiny edits, one-liners\n"
    "- reasoning: analysis, planning, comparisons, explanations, math\n"
    "- deep_code: complex implementation, debugging, refactoring, architecture\n"
    "- agent_work: multi-step repo work — run commands, read/edit files, tests, builds\n"
    "- vision: the request has attached images or screenshots to interpret\n"
    "- design_ui: anything about how software looks or feels — UI/UX, layouts, visual design, "
    "styling, CSS, colors, typography, design systems, mockups, front-end polish. Pick this "
    "whenever the subject is visual/design, even if the request is short, conversational, or "
    "also involves writing code.\n"
    "No other text, no markdown fences."
)

MAX_TEXT_CHARS = 2000
CACHE_LIMIT = 512


def parse_reply(text: str) -> str | None:
    """Best-effort category extraction: first balanced {...} that has a known category."""
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        cat = obj.get("category") if isinstance(obj, dict) else None
        if isinstance(cat, str) and cat.strip() in CATEGORIES:
            return cat.strip()
    return None


class Classifier:
    """Calls the z.ai chat/completions endpoint; caches decisions; never raises (falls back)."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_s: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.calls = 0            # observable in tests / logs
        self._ssl = ssl.create_default_context()

    def _ask(self, text: str, has_images: bool) -> str:
        prompt = text[:MAX_TEXT_CHARS]
        if has_images:
            prompt = "[images attached]\n" + prompt
        return json.dumps({
            "model": self.model, "temperature": 0, "max_tokens": 1024,
            "thinking": {"type": "disabled"},   # routing must be fast, not reasoned
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": prompt or "(empty)"}],
        }).encode()

    async def _post(self, body: bytes) -> dict[str, Any]:
        u = self.base_url.split("://", 1)[1]
        host, _, path = u.partition("/")
        tls = self.base_url.startswith("https://")
        port = 443 if tls else 80
        if ":" in host and not host.startswith("["):
            host, port_s = host.rsplit(":", 1)
            port = int(port_s)
        kwargs: dict[str, Any] = {"ssl": self._ssl} if tls else {}
        reader, writer = await asyncio.open_connection(host, port, **kwargs)
        try:
            head = (f"POST /{path}/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                    f"Authorization: Bearer {self.api_key}\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                    f"Connection: close\r\n\r\n").encode()
            writer.write(head + body)
            await writer.drain()
            raw = await reader.read()
        finally:
            writer.close()
        header, _, payload = raw.partition(b"\r\n\r\n")
        status = int(header.split(b" ", 2)[1])
        return {"status": status, "body": payload}

    async def classify(self, text: str, has_images: bool) -> tuple[str, bool]:
        """→ (category, from_cache). Falls back to "" on any failure."""
        fp = hashlib.sha256(f"{has_images}\x00{text}".encode()).hexdigest()
        hit = self._cache.get(fp)
        if hit is not None:
            self._cache.move_to_end(fp)
            return hit, True
        category = ""
        try:
            self.calls += 1
            resp = await asyncio.wait_for(self._post(self._ask(text, has_images)),
                                          timeout=self.timeout_s)
            if resp["status"] == 200:
                data = json.loads(resp["body"].decode("utf-8", "replace"))
                content = data["choices"][0]["message"]["content"]
                content = content if isinstance(content, str) else json.dumps(content)
                category = parse_reply(content) or ""
        except Exception:  # noqa: BLE001 — the router must route even when z.ai is down
            category = ""
        if category:
            self._cache[fp] = category
            while len(self._cache) > CACHE_LIMIT:
                self._cache.popitem(last=False)
        return category, False


def log_decision(log, category: str, cached: bool, target: str, took_ms: int) -> None:
    log.info("route %-11s → %-28s (%s, %dms)", category or "fallback", target,
             "cached" if cached else "classified", took_ms)
