"""The Jev classifier: one tiny TypeSafe System One call turns a request into a routing
category.

Jev returns a typed choice with calibrated probabilities — no prompt-JSON wrangling, no
fences to strip. Decisions are cached by (text, has_images) so a multi-turn conversation
— where the client resends the transcript — classifies once, not per turn.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import ssl
from collections import OrderedDict
from typing import Any

CATEGORIES = ("quick_chat", "quick_code", "reasoning", "deep_code", "agent_work", "vision",
             "design_ui")

# The choice criteria carry the full category definitions (per-criterion descriptions).
_INSTRUCTIONS = (
    "Classify the user's request to a coding-assistant router into exactly one routing "
    "category. Judge by the kind of work the request needs, not its length."
)
_CRITERIA = {
    "quick_chat": "Greetings, simple Q&A, facts, small talk.",
    "quick_code": "Short snippets, syntax questions, tiny edits, one-liners.",
    "reasoning": "Analysis, planning, comparisons, explanations, math.",
    "deep_code": "Complex implementation, debugging, refactoring, architecture.",
    "agent_work": "Multi-step repo work — run commands, read/edit files, tests, builds.",
    "vision": "The request has attached images or screenshots to interpret.",
    "design_ui": (
        "Anything about how software looks or feels — UI/UX, layouts, visual design, "
        "styling, CSS, colors, typography, design systems, mockups, front-end polish. "
        "Pick whenever the subject is visual/design, even if the request is short, "
        "conversational, or also involves writing code."
    ),
}

_DIFF_INSTRUCTIONS = (
    "How difficult is this request for the model that will handle it? Judge the reasoning "
    "depth, subtlety, and stakes — not the topic."
)
_DIFF_CRITERIA = {
    "light": (
        "Simple, well-defined, one obvious answer or a tiny change. A fast small model "
        "handles it comfortably."
    ),
    "standard": (
        "Normal everyday task — some context, nuance, or multi-part scope, but a "
        "competent mid-tier model handles it reliably."
    ),
    "hard": (
        "Genuinely difficult — subtle bugs, careful tradeoffs, architecture decisions, "
        "precise reasoning chains, unfamiliar territory, or high stakes. Deserves the "
        "strongest model available."
    ),
}

MAX_TEXT_CHARS = 2000
CACHE_LIMIT = 512

# Escalation gates — expensive routes require Jev to be reasonably sure.
# If probabilities are missing (older backends, fakes), treat the choice as confident.
HARD_DIFFICULTY_MIN_P = 0.60     # "hard" below this → standard (mid-tier model)
DEEPCODE_MIN_P = 0.50           # deep_code below this → reasoning (capable, cheaper)


def parse_answer(data: Any) -> tuple[str, str, float, float] | None:
    """→ (category, difficulty, p_cat, p_diff) from a System One body, or None if unusable.
    Difficulty defaults to "standard" when missing or unknown; probabilities default to 1.0."""
    try:
        cat_answer = data["answers"]["category"]
        cat = cat_answer.get("choice") if isinstance(cat_answer, dict) else None
        if not (isinstance(cat, str) and cat in CATEGORIES):
            return None
        p_cat = float(cat_answer.get("probabilities", {}).get(cat, 1.0))
        diff_answer = data["answers"].get("difficulty")
        diff = diff_answer.get("choice") if isinstance(diff_answer, dict) else None
        if not (isinstance(diff, str) and diff in _DIFF_CRITERIA):
            diff = "standard"
        p_diff = float(diff_answer.get("probabilities", {}).get(diff, 1.0)) \
            if isinstance(diff_answer, dict) else 1.0
        return cat, diff, p_cat, p_diff
    except (KeyError, TypeError, ValueError):
        return None


def calibrate(category: str, difficulty: str, p_cat: float,
              p_diff: float) -> tuple[str, str, str]:
    """Apply the escalation gates. → (category, difficulty, note).

    A hesitant "hard" or a hesitant "deep_code" must not buy the strongest model:
    downgrade to the tier below unless Jev is confident enough.
    """
    note = ""
    if category == "deep_code" and p_cat < DEEPCODE_MIN_P:
        category = "reasoning"
        note = f"deep_code p={p_cat:.2f}→reasoning"
    if difficulty == "hard" and p_diff < HARD_DIFFICULTY_MIN_P:
        difficulty = "standard"
        note = (note + "; " if note else "") + f"hard p={p_diff:.2f}→standard"
    return category, difficulty, note


class Classifier:
    """Calls the TypeSafe System One endpoint; caches decisions; never raises (falls back)."""

    def __init__(self, *, base_url: str = "https://api.typesafe.ai", api_key: str,
                 model: str = "jev-latest", timeout_s: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.calls = 0            # observable in tests / logs
        self._ssl = ssl.create_default_context()

    def _ask(self, text: str, has_images: bool) -> bytes:
        state = text[:MAX_TEXT_CHARS]
        if has_images:
            state = "[images attached]\n" + state
        return json.dumps({
            "state": state or "(empty)",
            "model": self.model,
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": _INSTRUCTIONS,
                    "criteria": _CRITERIA,
                },
                "difficulty": {
                    "type": "choice",
                    "instructions": _DIFF_INSTRUCTIONS,
                    "criteria": _DIFF_CRITERIA,
                },
            },
        }).encode()

    async def _post(self, body: bytes) -> dict[str, Any]:
        u = self.base_url.split("://", 1)[1]
        host, _, path = u.partition("/")
        target = f"/{path}/v1/systemone" if path else "/v1/systemone"
        tls = self.base_url.startswith("https://")
        port = 443 if tls else 80
        if ":" in host and not host.startswith("["):
            host, port_s = host.rsplit(":", 1)
            port = int(port_s)
        kwargs: dict[str, Any] = {"ssl": self._ssl} if tls else {}
        reader, writer = await asyncio.open_connection(host, port, **kwargs)
        try:
            head = (f"POST {target} HTTP/1.1\r\nHost: {host}\r\n"
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

    async def classify(self, text: str, has_images: bool) -> tuple[str, str, bool]:
        """→ (category, difficulty, from_cache). Falls back to ("", "", False) on failure."""
        fp = hashlib.sha256(f"{has_images}\x00{text}".encode()).hexdigest()
        hit = self._cache.get(fp)
        if hit is not None:
            self._cache.move_to_end(fp)
            category, _, difficulty = hit.partition("\x00")
            return category, difficulty, True, ""
        category, difficulty = "", ""
        note = ""
        try:
            self.calls += 1
            resp = await asyncio.wait_for(self._post(self._ask(text, has_images)),
                                          timeout=self.timeout_s)
            if resp["status"] == 200:
                data = json.loads(resp["body"].decode("utf-8", "replace"))
                parsed = parse_answer(data)
                if parsed:
                    category, difficulty, p_cat, p_diff = parsed
                    category, difficulty, note = calibrate(category, difficulty,
                                                           p_cat, p_diff)
        except Exception:  # noqa: BLE001 — the router must route even when TypeSafe is down
            category, difficulty = "", ""
        if category:
            self._cache[fp] = f"{category}\x00{difficulty}"
            while len(self._cache) > CACHE_LIMIT:
                self._cache.popitem(last=False)
        return category, difficulty, False, note


def log_decision(log, category: str, difficulty: str, cached: bool, target: str,
                 took_ms: int, note: str = "") -> None:
    extra = f"; {note}" if note else ""
    log.info("route %-11s %-9s → %-28s (%s, %dms%s)", category or "fallback",
             difficulty or "-", target, "cached" if cached else "classified",
             took_ms, extra)
