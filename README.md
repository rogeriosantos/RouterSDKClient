# RouterSDKClient — warm router

A thin OpenAI-compatible router in front of the warm stack: **Jev (TypeSafe System One)
classifies every `router/auto` request** by category and difficulty in one call, then the gateway
proxies it byte-for-byte to the right backend — the z.ai API, the claude-warm gateway
(`rogeriosantos/ClaudeSDKClient`), or the codex-warm gateway
(`rogeriosantos/CodexSDKClient`).

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
python -m router_warm.gateway --port 8795
```

## Routing

| Requested model | Behaviour |
| --- | --- |
| `router/auto` | one Jev call (`jev-latest`, two typed choices, cached per message) classifies the last user message; a category × difficulty table picks the target |
| `glm-*` | passthrough → z.ai API (`api.z.ai/api/coding/paas/v4`) |
| `claude-*` | passthrough → claude-warm `:8793` |
| `gpt-*` | passthrough → codex-warm `:8794` |

Default table (override cells in `~/.config/router-warm/config.json`; restart to reload):

| Category | Light | Standard | Hard |
| --- | --- | --- | --- |
| `quick_chat` | GLM 5.3 Flash | GLM 5.3 | GLM 5.3 |
| `quick_code` | GLM 5.3 Flash | GLM 5.3 | Claude Sonnet 5.5 |
| `reasoning` | GLM 5.3 | GPT-6 Sol | GPT-6 Sol |
| `deep_code` | Claude Sonnet 5.5 | Claude Sonnet 5.5 | Claude Opus 5.5 |
| `agent_work` | GPT-6 Luna | GPT-6 Luna | Claude Opus 5.5 |
| `vision` | GLM 5.3 Flash | GPT-6 Luna | GPT-6 Astra |
| `design_ui` | GLM 5.3 | GPT-6 Astra | Claude Opus 5.5 |

GLM 5.3 is text-only. A request with current-turn media that would otherwise reach it
moves to the matching `vision` difficulty route. Classifier failures use
`fallback` (`claude-sdk/claude-opus-5-5`).

## Endpoints & security

`GET /health`, `GET /v1/models`, `GET /v1/status`, `POST /v1/chat/completions` (stream +
non-stream, proxied untouched — status line, headers, SSE and all). `/v1/status` is the
live task tracker: in-flight requests (phase, category/difficulty, backend/model, elapsed,
first-byte time, bytes), the last 20 finished, and failover/error counters. `router-status`
(`~/.pi/agent/bin/router-status`, `-w` to watch) pretty-prints it. While a backend thinks,
the log emits a waiting beacon every 30s (`req <id>: waiting 60s for first byte from …`).
Binds 127.0.0.1; bearer key in
`~/.config/router-warm/gateway.key` (created 0600 on first start); browser `Origin`s are
rejected. `X-Pi-Cwd` is forwarded to the two agent gateways. A client disconnect tears the
backend connection down. z.ai credentials come from `ZAI_API_KEY` or
`~/.config/router-warm/zai.key`; the Jev key comes from `TYPESAFE_API_KEY` or
`~/.config/router-warm/jev.key`; the gateway keys of the two warm backends are read from
their own `~/.config/{claude,codex}-warm/gateway.key` files.
