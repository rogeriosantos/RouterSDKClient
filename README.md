# RouterSDKClient — warm router

A thin OpenAI-compatible router in front of the warm stack: **z.ai (GLM) classifies every
`router/auto` request**, then the gateway proxies it byte-for-byte to the right backend —
the z.ai API, the claude-warm gateway (`rogeriosantos/ClaudeSDKClient`), or the codex-warm
gateway (`rogeriosantos/CodexSDKClient`).

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
python -m router_warm.gateway --port 8795
```

## Routing

| Requested model | Behaviour |
| --- | --- |
| `router/auto` | one tiny `glm-5.3-flash` call (thinking disabled, ~2s, cached per message) classifies the last user message; a category→backend table picks the target |
| `glm-*` | passthrough → z.ai API (`api.z.ai/api/coding/paas/v4`) |
| `claude-*` | passthrough → claude-warm `:8793` |
| `gpt-*` | passthrough → codex-warm `:8794` |

Default table (edit `~/.config/router-warm/config.json`, no restart needed to change — it is
read at startup; restart to reload):

- `quick_chat` / `quick_code` / `reasoning` → `zai/glm-5.3`
- `deep_code` / `agent_work` → `claude-sdk/claude-opus-5-5`
- `vision` / `design_ui` → `codex-sdk/gpt-6-astra` — everything visual: screenshots, UI/UX, styling, layouts
- classifier unreachable → `fallback` (`claude-sdk/claude-opus-5-5`) — the router never becomes the outage

## Endpoints & security

`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (stream + non-stream, proxied
untouched — status line, headers, SSE and all). Binds 127.0.0.1; bearer key in
`~/.config/router-warm/gateway.key` (created 0600 on first start); browser `Origin`s are
rejected. `X-Pi-Cwd` is forwarded to the two agent gateways. A client disconnect tears the
backend connection down. z.ai credentials come from `ZAI_API_KEY` or
`~/.config/router-warm/zai.key`; the gateway keys of the two warm backends are read from
their own `~/.config/{claude,codex}-warm/gateway.key` files.
