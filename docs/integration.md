# Connecting apps

`origami serve` answers on `http://127.0.0.1:11439` (the port is the `port` setting; `origami status` shows the one
in use). It speaks the OpenAI, Anthropic and Ollama APIs, all translated onto the same chat path, so tools, images,
thinking and streaming work in each of them.

One request runs at a time; the others wait in a queue.

## Endpoints

| Endpoint | What it is |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/messages/count_tokens` | Anthropic's token count: `input_tokens` as the same body would be read (template, system, tools), no reply |
| `POST /v1/responses` | OpenAI Responses (no `previous_response_id`: send the whole conversation) |
| `POST /api/chat`, `POST /api/generate` | Ollama (`/api/generate` without `raw` or `suffix`) |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama model list and details |
| `GET /v1/models`, `GET /v1/models/<id>` | the model list (`?all` lists every model) |
| `GET /health` | `loading` (with `loading.seconds` and the model's last log line, `loading.line`), `ready`, `degraded` (the model's drive went away: `degraded` says what to do; it clears by itself when the drive is back) or `failed`, the queue and the server defaults; `stopping` or `stopped` (with a 503) after `origami stop`; `pending` while a reload waits for the current reply; `slowed_by` as in `/v1/status` |
| `GET /v1/status` | memory (`temporarily_reduced` while the budget is below `recovery_target_bytes`, the most it may use, which it climbs to, in either mode; `last_cut`: the last cut's reason and time; `learned_start_bytes`: automatic mode's start, learned from this Mac's memory use, which it grows past as memory frees), the running request, recent requests, saved prompts, drives and the last 50 log lines (`events`), and `slowed_by`, as JSON |
| `GET /v1/plan?context=&memory_limit=&expert_memory=&hot_copy=&context_dir=&quality=` | a preview of the memory plan for those settings |
| `POST /v1/config` | saves settings for the next start (used by the dashboard) |
| `POST /v1/stop`, `POST /v1/start` | the same as `origami stop` and `origami start` |
| `POST /v1/jobs`, `GET` (progress, or the last job's outcome), `DELETE` (cancel or dismiss) | dashboard jobs: add a model from a folder, file or Hugging Face repo |
| `/v1/hot-copy`, `/v1/prompts`, `/v1/memory/learned`, `/v1/memory/wired`, `/v1/storage`, `/v1/fs`, `/v1/reload`, `/v1/restart`, `/v1/lan`, `/v1/tuning/apply`, `/v1/config/restore` | the dashboard's own controls (hot copy, saved prompts, memory, files, reload, access, tuning); not a stable API |
| `GET /v1/models/found`, `GET /v1/models/check?source=` | models found on this Mac, and whether a source would run |
| `GET /dashboard` (and `GET /` from a browser) | the dashboard (see [dashboard.md](dashboard.md)) |

`slowed_by` in `/health` and `/v1/status` lists what slows Origami down now, the costliest first and advice last, as
the dashboard's Now tab and `origami status` show it: `[{"id", "text", "fix", "since"}]`, empty when nothing does.
`text` says what, `fix` what to do (or that it recovers by itself, and when), `since` (ISO 8601) when it began or was
first seen. The ids: `drive_gone`, `job`, `memory_low`, `memory_limit_not_kept`, `model_drive_slow`, `model_drive_due`,
`drive_not_allowed`, `copy_drive_gone`, `hot_copy_building`, `hot_copy_no_room`, `hot_copy_failed`, `hot_copy_short`,
`hot_copy_drive_slow`, `mac_hot`, `low_power`, `saves_paused`, `prompt_read_again`, `queue`, `memory_limit_low`,
`not_tuned`; more may come, so show `text` and `fix` rather than matching ids. `/v1/status` also carries what the list reads from this Mac: `mac_now` (`thermal_state`, 0 to 3 as macOS reports it, and `low_power`) and, while the memory budget is 1 GB or more under what Origami may use (`memory.target_bytes`), `memory_holders` (the three largest other apps, `[{"name", "bytes"}]`).

In `/v1/status`, `experts_from.sources` lists where experts are read from; the model's own row (`"role": "all"`), while
its files are read, carries `slow_reads`: `busy_seconds` and `slow_seconds` (reading time since start, and the part of it
below a third of the drive's own recent best rate), `share`, the same for the last hour (`hour_busy_seconds`,
`hour_slow_seconds`, `hour_share`), `healthy_bytes_per_second`, `threshold` (0.2) and `high` (over the threshold with at
least 60 s of reading). While `high`, the row's `detail` says to run `origami refresh` (see
[troubleshooting.md](troubleshooting.md#the-model-drive-got-slow)). The model's row and the hot copy's carry
`since_rewrite` once anything was read from them: what Origami read from those files since they were last written,
kept across restarts in `<data_dir>/drive-reads.json` (`bytes`, `busy_seconds`, `slow_seconds`, `share`, `since`,
`rewritten`), `last_slow_after_bytes` (how much had been read when the slowdown began last time, or null), `high`,
`due` (80% of that read again, and not slow yet) and `text`, one sentence that says it.

The hot copy's row (`"role": "hot"`) carries `budget_bytes` (its size limit), `coverage` (once replies are counted: a
list of `{"bytes", "share"}`, the share of counted expert uses a copy of that size holds, for the size limit and each
size the drives offer), `size_limit` (what set an automatic size: `free_space`, `flat`, `whole_set` or `no_counts`;
null for a size set by you) and `size_note` (the same in words). `/v1/plan?hot_copy=` gives the share for the size asked
(GLM and DeepSeek: `hot_copy.coverage`; Qwen: `experts.hot_copy_coverage`), null before replies are counted.

Each `/v1/models` entry (and `/api/tags`, `/api/show`) carries what apps look for: `reasoning_efforts`,
OpenRouter's `supported_parameters`, `context_length`, and Ollama's `capabilities` (`thinking`, `tools`, and `vision`
for a model that reads images).

By default the list holds one model, `origami`, and every request gets the model chosen in the dashboard whatever
name it sends. With `list_models true` it lists every model and a request can switch between them by name; see
[models.md](models.md#keeping-several-models).

A switch costs a full stop and load: the running reply finishes, the model's memory goes back, the other model loads
(GLM and DeepSeek take about a minute to answer again, Qwen less) and starts with an empty expert cache, so its first
replies are slower until the cache refills; its saved prompts are kept. Alternating two models on every request pays
that each time. To soften it, requests for the loaded model that wait together go first: up to four of them go ahead
of a waiting request for another model, and none once that one has waited 15 minutes, so it is never starved. Send a
batch to one model, then the next, rather than interleaving them.

Not supported: Responses' `previous_response_id`, server-side tools (web search and the like are left out), and
Responses' `custom` tools (free-form input; refused with a 400 that names the tool: declare it as a function). A
Responses `reasoning` input item is the next assistant turn's reasoning, as `reasoning_content` is on the chat wire.
An Anthropic `tool_result` with images passes them to a model that reads images (as a user turn right after the
results); for one that does not, the result says an image was left out. A reply ended by a `stop` string says so on
the Anthropic wire (`stop_reason` `stop_sequence`, with the string); the other wires say `stop`. A model name selects
a quality level only as this model's id (or `origami`) and `-<level>`, such as `qwen3.8-flash-next-faster`.

## Requests

- **Usual fields.** Streaming, tool calls, `stop`, `seed` and the usual sampling fields (`temperature`, `top_p`,
  `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`; Ollama's `options` carry the same names); `top_p: 0`
  means greedy. Each model's tool calls are read in its own syntax (Qwen's, GLM-5.3's, DeepSeek-V4.1's). A call to a
  tool the request did not declare, or one the model wrote badly, comes back as the model's text with the usual finish
  reason. A streamed call is sent once its block is read whole, so a badly written one streams only as text (text
  before a call streams as it comes).
- **Tools.** `strict` and other keys Origami does not use are ignored: the schema is shown to the model, not
  enforced. Chat control markers typed into a message or tool result stay text. `tool_choice` `required` (Anthropic
  `any`) or a named tool adds one sentence to the end of the last user message, so the conversation before it still
  reuses saved state, and the reply is made to open a tool call (that tool's, when named), also beside JSON mode. A
  named choice keeps every declared tool in the prompt. A reply that still holds no valid call (one the model wrote
  badly) is a 422 `tool_choice_unmet`; one cut by `max_tokens` finishes as `length`. `none` leaves the tools out.
- **Limits.** Bodies up to 24 MB, 512 tools and 4,096 messages a request. A `max_tokens` larger than the context
  leaves is lowered to fit. A prompt longer than the context is refused in the client's own wording (for Anthropic:
  "prompt is too long: N tokens > M maximum").
- **JSON mode.** `response_format` `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema":
  …}}`, the Responses `text.format` (`json_object`, `json_schema`) and Ollama's `format` (`"json"` or a schema) make
  the answer one valid JSON value: each token is chosen only among those that keep it valid, so the model still
  decides what to write. `json_object` asks for an object; a schema is enforced whether `strict` is true or not.
  Supported in a schema: `type` (one or a list), `properties`, `required`, `additionalProperties`, `items`,
  `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `$ref` into the schema itself (`#/$defs/…`, recursion included)
  and annotations such as `description`, `title`, `default` and `format` (not enforced). Any other keyword
  (`pattern`, `minLength`, `minimum`, `oneOf`, …) is refused with a 400 that names it. `enum` and `const` take `type` beside them (it narrows the values) and nothing else that constrains;
  an object key is never matched through an escape, so an extra key cannot spell a declared one (`\u0078` for `x`).
  The schema is not shown to the model: describe the answer in the prompt too. With thinking on, the model thinks freely and the answer after
  it is the JSON; with tools declared the reply may still be a tool call instead (with `tool_choice` required or
  named it must be one). The answer starts with the JSON
  (no text or spaces before it) and ends with it. Whitespace inside it is limited to a newline and an indent. A
  reply cut by `max_tokens` (`finish_reason` `length`) or by a `stop` string holds incomplete JSON. Prompts reuse
  saved state as usual; the constraint is not saved with it.
- **Queue.** One request runs at a time; up to 64 others wait, a person's message ahead of an agent's tool turn.
  Memory for request bodies is sized for 4 large requests at once (2 at a 256K context); when it is full a new
  request gets a 503 "Request buffers are busy; retry in a moment".
- **Keepalive.** A streamed reply (server-sent events) sends a `: keepalive` comment every 10 s while it waits or
  reads a long prompt. Ollama's NDJSON streams do not. Set client timeouts to 10 minutes or more.
- **Closing early.** A client that closes its sending side before the reply on a non-streamed request is treated as
  gone, and its request is cancelled; use streaming, or keep the connection open.

### Images

Qwen reads images; GLM and DeepSeek refuse requests with images.

- Send OpenAI `image_url` parts in user messages, as base64 `data:` URLs or http(s) links. PNG, JPEG, HEIC, GIF,
  WebP or TIFF, up to 24 MB and about 67 megapixels.
- At most 32 images are read per request, counting only those not already in saved state, so a long chat can hold
  more.
- Links are fetched within a minute and may redirect, but never from https to http, and never to this Mac or a
  private or link-local network address (checked after name resolution and on every redirect; such a link is
  refused with `image_link_refused`). Send such an image as a `data:` URL instead. Local files are not read.
- An image is scaled to at most about one megapixel and becomes up to 1,024 tokens. The vision part takes 1 to 4 s
  per image on an M2 Pro and borrows about 2 GB from the expert cache while it runs.
- The next turn of a chat continues after its images without reading them again, when they are sent unchanged.

### Thinking

Every model thinks, and a request can choose how much:

- `reasoning_effort`: `none` or `minimal`, `low`, `medium`, `high`, `xhigh`, `max`;
- or the fields harnesses send instead: `think`, OpenRouter's `reasoning: {effort, enabled, max_tokens, exclude}`,
  Anthropic's `thinking: {type, budget_tokens}`, `chat_template_kwargs.enable_thinking`;
- a budget: `max_thinking_tokens` (or `thinking_budget`).

Each model's chat template tells fewer levels apart; every name is accepted and `/v1/models` lists each distinct
level once (`reasoning_efforts`):

| Model | Distinct levels | The same as |
|---|---|---|
| Qwen3.8-Flash-Next | `none`, `low`, `medium`, `high` | `minimal` = `none`; `high` = `xhigh` = `max` (the template's xhigh) |
| GLM-5.3-Flash | `none`, `low`, `high`, `max` | `minimal` = `none`; `medium` = `xhigh` = `max` (the template knows only Low, High, Max) |
| DeepSeek-V4.1-Flash | `none`, `low`, `medium`, `high`, `max` | `minimal` = `none`; `xhigh` = `max` |

A request that sends none gets the server's `reasoning_effort` setting (default `none`). When thinking would leave
less than max(1024, 15% of `max_tokens`) for the answer (capped at half of `max_tokens`), Origami closes the think
block so the reply still ends with an answer (`usage.completion_tokens_details.reasoning_closed_by` says so).
`/health` lists every thinking field.

### Saved prompts

Repeated prompts, chat follow-ups and a harness's system prompt reuse saved prompts, also after a restart.
`usage.prompt_tokens_details.cached_tokens` shows how much.

- Keep dates and other changing text at the end of a system prompt: everything before the first change is reused.
- Send `"prompt_cache_policy": "no_store"` for one-off requests (titles, summaries), so they do not push out
  useful entries.
- Tool-call follow-ups can reuse a checkpoint at the first tool-call marker, even when a client rewrites the
  arguments. Only an identical token prefix is reused; changed earlier reasoning is read again.
- Reused state can round slightly differently from a cold run; see [settings.md](settings.md#which-settings-change-replies).

### Keeping a session

A client that returns to one long conversation between other chats can send the header `X-Origami-Keep: <name>`
(letters, digits, `.`, `_`, `-`). After the reply, that conversation's saved prompt stays pinned under the name and
moves to each newer turn, so other chats cannot push it out. Two sessions at most: a third releases the one used
longest ago. `origami prompts` lists them and `origami forget <name>` releases one. Prompts under 512 tokens and
`no_store` requests are not saved, so they cannot be kept.

### Progress while a long prompt is read

With `origami config set status_messages true` (or `"status_messages": true` in a request), a streamed reply shows
short `Origami ·` lines in the thinking panel while it waits or reads a long prompt: how much of the prompt is read
and how much came from saved state. Clients that send the thinking back get those lines dropped.
`"return_progress": true` sends llama.cpp-style `prompt_progress` frames instead.

### A time limit

`"origami_time_limit": <seconds>` bounds a request from its arrival. Past it, the reply ends after the token being
produced, with the text so far and finish reason `length` (Anthropic `max_tokens`), and the conversation continues
from there as after `max_tokens`. A limit that passes while the prompt is still being read gets a 408
`request_timeout`; the part already read is kept, so sending the request again continues from it.

## Chat commands

Type a command as the whole message in any chat app or agent. Origami answers it without running the model, and
drops earlier commands and their answers from what the model sees.

| Command | What it does |
|---|---|
| `origami help` | lists the commands |
| `origami status` | version, model, memory, stores, saved prompts |
| `origami save [name]` | keeps this chat's system prompt cached, pinned under a name (letters, digits, `.`, `_`, `-`; not `all`) |
| `origami prompts` | the saved system prompts and kept sessions |
| `origami forget <name\|all>` | removes saved system prompts and kept sessions |
| `origami quality [level]` | the quality level for chats with this system prompt (a request's own choice wins); alone, shows the current one |
| `origami fast on\|off` | the same as `origami quality faster` or `origami quality exact` |

Anything else, such as `origami help?` or a question about origami, is an ordinary message.
`origami config set commands false` turns commands off.

## Stop and start

`origami stop` (or **Stop and unload** in the dashboard) stops the model and gives all its memory back to macOS. The
server keeps running, so the dashboard still opens. A request still being answered gets 30 s to finish, then it is
cancelled.

Until you start it again, apps get a 503 with `"code": "stopped"` and `X-Should-Retry: false` (so the OpenAI SDKs do
not retry). It stays stopped across a restart of the server or the Mac. `origami start` (or **Start**) loads the same
model again and waits until it is ready. `POST /v1/stop` and `POST /v1/start` do the same over HTTP (with the API key
when one is set, and `Content-Type: application/json`).

Two opt-in settings change that. With `origami config set start_on_request true`, a request to a stopped server
starts the model (as `origami start` does) and waits for it instead of getting the 503. With
`origami config set idle_unload_minutes 30`, the model unloads after 30 minutes without a request, exactly as a stop
does (saved, so it stays unloaded across a restart); together with `start_on_request` the next request loads it again.

## Apps and harnesses

The dashboard's **Connect an app** (in the Connect tab) shows the base URL, the model id and a ready-made Pi block with this server's
values, with copy buttons.

**Open WebUI.** Add an OpenAI connection with URL `http://127.0.0.1:11439/v1` and any key.

**Ollama CLI.** `OLLAMA_HOST=127.0.0.1:11439 ollama run qwen3.8-flash-next` (or any name `ollama list` shows);
`ps` and `show` work too.

**Coding agents need a bigger context.** Claude Code and Codex send a 7K to 11K-token system prompt, so set
`origami config set context 32k` (or more). Their first request reads that prompt (about 100 s on an M2 Pro); later
ones start from the saved copy. Tested with Claude Code 2.1 and Codex 0.159 (a question and a file-reading tool turn
each), the official Anthropic and OpenAI Python SDKs, and raw Ollama requests.

**Claude Code:**

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:11439 ANTHROPIC_API_KEY=origami ANTHROPIC_MODEL=qwen3.8-flash-next \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 claude
```

The context variable tells Claude Code the real window, so it compacts before the window fills (not tried in a long
session). Without the output variable it asks for 32,000 tokens, more than a 32K window leaves; Origami lowers it to
what fits.

**Codex.** In `~/.codex/config.toml`:

```toml
model = "qwen3.8-flash-next"
model_provider = "origami"

[model_providers.origami]
base_url = "http://127.0.0.1:11439/v1"
wire_api = "responses"
env_key = "ORIGAMI_KEY"            # any value
stream_idle_timeout_ms = 1800000
```

Without an OpenAI login, Codex 0.159's startup call to chatgpt.com hung until `check_for_update_on_startup = false`,
`[analytics] enabled = false`, and `[features] remote_models = false` and `plugins = false` were set.

**Pi** reads a model's thinking levels only from its own `~/.pi/agent/models.json`. With this, `/model` lists
`origami` and `/thinking` offers every level:

```json
{
  "providers": {
    "origami": {
      "baseUrl": "http://127.0.0.1:11439/v1",
      "api": "openai-completions",
      "apiKey": "origami",
      "compat": { "supportsReasoningEffort": true, "thinkingFormat": "openai", "maxTokensField": "max_tokens" },
      "models": [{
        "id": "origami", "name": "Origami", "reasoning": true,
        "thinkingLevelMap": { "off": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max" },
        "input": ["text", "image"], "contextWindow": 8192, "maxTokens": 4096
      }]
    }
  }
}
```

Set `contextWindow` to the served context (`origami config get context`), `input` to `["text"]` for GLM and
DeepSeek, and `apiKey` to `"$ORIGAMI_API_KEY"` on a server with a key. `maxTokens` is sent with every request and
must leave room for the prompt, so keep it well under the context.

**llama-swap** (one entry per context):

```yaml
models:
  qwen3.8-flash-next:
    cmd: origami serve --port ${PORT} --context 8k
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    ttl: 600
```

To keep separate settings per entry, give each its own settings file with `--config <file>`.

**Start at login (LaunchAgent).** `origami service install` does all of this: it writes a plist like the one below for the
binary you run it with (with the setting options you give it, such as `--port 11500 --model glm`, folders made absolute,
and the `ORIGAMI_` variables of your shell), prints it, creates the log folder and loads it;
`origami service remove` stops and deletes it (`--label` names another service; a different plist of that label is
replaced only with `--yes`). By hand: save as `~/Library/LaunchAgents/com.origami.serve.plist`, with the paths
changed to where you built `origami` (launchd needs the full path) and to your home folder:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.origami.serve</string>
  <key>ProgramArguments</key><array><string>/Users/YOUR_NAME/origami/.build/release/origami</string><string>serve</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/Users/YOUR_NAME/Library/Logs/Origami/origami.log</string>
  <key>ExitTimeOut</key><integer>180</integer>
</dict></plist>
```

Create its log folder and load it (launchd creates a missing log folder itself, but readable by every user of the Mac;
checked on macOS 26 with a test label):

```sh
mkdir -p ~/Library/Logs/Origami
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.origami.serve.plist
```

Stop it (until the next login or `bootstrap`) with `launchctl bootout gui/$(id -u)/com.origami.serve`. `ExitTimeOut` gives
a long conversation's save time to finish at logout (launchd's default is 20 s). In Terminal, Ctrl-C waits for saves;
a second Ctrl-C quits at once.

If the model or a store is on an external drive, macOS may need to allow **Removable Volumes** for `origami` (or
your launcher); see [troubleshooting.md](troubleshooting.md).

**Keeping that permission across upgrades.** macOS stores the permission for the service's own program, tied to that
exact file, so a new `origami` binary as the service's program is asked about again. A small fixed launcher avoids
it: build `Tools/launcher/origami-launcher.c` once (`cc -O2 -o origami-launcher Tools/launcher/origami-launcher.c`),
keep the built file where it is (never rebuild or move it), and make it the service's program with `origami` as its
first argument: `ProgramArguments` = `/path/to/origami-launcher`, `/path/to/origami`, `serve`. macOS then asks once,
for the launcher; an upgrade replaces only `origami`. The launcher starts origami, passes on the signals that stop it
and ends as origami ended, so `KeepAlive` still sees a crash as a crash. Origami's messages name the program macOS
asks about.

## Request records

Off by default. `origami config set request_log ~/origami-requests` writes one JSON line per request (API, client,
token counts, cache share, timings, how it finished) and each distinct system prompt and tool list once under
`units/`, readable by you only, kept 30 days. Useful to see what a harness really sends.

## Security

Origami listens on 127.0.0.1 only (unless you set `listen lan`) and answers only requests whose `Host` is a loopback
address. Requests from web pages on other sites (and from opaque `Origin: null` pages) get a 403, so a page you
visit cannot drive it.

Without an API key (the default), any program on this Mac that can reach the port can use it and change its
settings (`POST /v1/config`). With one, every request must carry it:

```sh
origami config set api_key "$(openssl rand -hex 16)"   # 16 to 256 characters; `origami config set api_key off` removes it
curl -H "Authorization: Bearer <key>" http://127.0.0.1:11439/v1/models    # or the header x-api-key: <key>
```

- A request without the key, or with another one, gets a 401 (`invalid_api_key`) before its body is read, on every
  endpoint, `/health` included (give a health checker the header).
- The key lives in the settings file (readable by you only) or `ORIGAMI_API_KEY`, never on a command line.
  `origami config`, `status` and the dashboard show only whether it is set.
- `origami status`, `bench`, `report` and `doctor` send it, and `origami dashboard` opens the page with it.
- The key is compared in constant time. Another device (or, behind a [TLS proxy](#tls), one of its clients) that
  sends 10 wrong keys within a minute gets a 429 until that minute ends; programs on this Mac are never locked out.
- Do not expose the port through a proxy or tunnel without the key or other authentication.

Slow or stuck clients cannot hold the server: the request line and headers must arrive within 10 s and the whole
request within 30 s, a body must keep arriving at 16 KB per 5 s at least (408 otherwise), and a large upload that is
too far behind to finish in time gives its buffer to another request that needs it (503). A client that stops reading
a reply is dropped after 30 s without taking a byte, and its request is cancelled as if it had disconnected (the
conversation is kept); the model never waits on it meanwhile. Another device may have at most 8 connections whose
request has not fully arrived yet, and all other devices together 32.

## Other devices on your network

To let an iPad, a phone or another computer in: `origami config set listen lan`, or **Who can reach Origami** in the dashboard's Connect tab. It
takes effect when Origami restarts (`origami serve` binds its port once; the dashboard restarts a launchd service for
you). The front then listens on every interface; the model's own server stays on 127.0.0.1.

```sh
origami config set listen lan     # prints http://<name>.local:11439/dashboard#key=<key>
```

- Other devices must send a key on every request (the dashboard page and its icons load without it). With no
  `api_key` set, Origami makes one for them, `lan_key`, so programs on this Mac keep working without a key. With
  `api_key` set, other devices send that.
- The dashboard on this Mac shows the address with the key and a QR code under **Who can reach Origami** (never on another device).
  Scan it, or open the address and paste the key when the page asks (it is kept in that browser).
- `Host` must be this Mac's `<name>.local`, its bare name or one of its addresses, so DNS rebinding still fails.
- `origami config set lan_key off` lets every device on the network in without a key: less safe, since anyone on
  your Wi-Fi can then use Origami and change its settings. `on` makes a new key.
- Traffic is plain HTTP on your network, the key included: anyone who can watch your Wi-Fi can read it
  (`origami doctor` warns). For TLS, put a proxy in front instead ([below](#tls)); from elsewhere, use a VPN. With the
  macOS firewall on, allow incoming connections for `origami` when macOS asks.

## TLS

Origami does not speak TLS itself. To keep the key off the network in clear, run a TLS proxy on this Mac in front of
it, and leave Origami on this Mac only:

```sh
origami config set listen local
origami config set api_key "$(openssl rand -hex 16)"   # every client sends it, through the proxy too
origami config set trusted_proxy loopback              # the proxy names each client (below)
```

`trusted_proxy loopback` makes a request from 127.0.0.1 that carries `X-Forwarded-For` count as coming from the
client the header's last entry names: it then needs the key (with no `api_key` it is refused), cannot use
`origami_tune`, is never shown the key under **Access**, and wrong keys are counted per client. A malformed header
gets a 400. The header is ignored from any other address, and without the setting (the default). The proxy must set
it (both recipes below do), or its clients are treated as programs on this Mac.

A certificate for this Mac's name (`scutil --get LocalHostName` prints the name before `.local`):

```sh
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 825 \
  -subj "/CN=mymac.local" -addext "subjectAltName=DNS:mymac.local" -keyout key.pem -out cert.pem
openssl x509 -noout -fingerprint -sha256 -in cert.pem   # compare this on each device
```

Give each device `cert.pem` to trust (an iPhone or iPad: AirDrop it, install the profile, then turn it on under
Settings → General → About → Certificate Trust Settings; curl: `--cacert cert.pem`) and check that the fingerprint it
shows is the one printed here. Clients that pin a certificate pin this one.

**Caddy** (`caddy run --config Caddyfile`):

```caddy
mymac.local:8443 {
	tls /path/to/cert.pem /path/to/key.pem
	reverse_proxy 127.0.0.1:11439 {
		header_up Host 127.0.0.1:11439
		header_up Origin ^https://mymac\.local:8443$ http://127.0.0.1:11439
		request_buffers 24MB
		flush_interval -1
	}
}
```

**nginx** (inside `http { … }`):

```nginx
map $http_origin $origami_origin { "https://mymac.local:8443" "http://127.0.0.1:11439"; default $http_origin; }
server {
    listen 8443 ssl;
    server_name mymac.local;
    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    client_max_body_size 24m;
    location / {
        proxy_pass http://127.0.0.1:11439;
        proxy_http_version 1.1;
        proxy_set_header Host 127.0.0.1:11439;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header Origin $origami_origin;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

Why each line: Origami answers only a loopback `Host`, so the proxy rewrites it; the dashboard saves settings only from
its own page, so the page's https origin is rewritten to the loopback one (and only that origin); Caddy sets
`X-Forwarded-For` to the client's address by itself, nginx is told to (replacing any value the client sent); bodies
are buffered by the proxy (a slow upload then costs Origami nothing) and replies are not, so streams arrive as they
are written; a request may wait up to an hour for the model. Devices then open
`https://mymac.local:8443/dashboard` and send the key as before.
