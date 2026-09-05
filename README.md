# model-studio-mcp

[![check](https://github.com/shakilmahmud-posh/model-studio-mcp/actions/workflows/check.yml/badge.svg)](https://github.com/shakilmahmud-posh/model-studio-mcp/actions/workflows/check.yml)

**A CLI and an MCP server for Alibaba Cloud Model Studio (DashScope), in stdlib-only Python.**
No SDK, no venv, no `pip install`. Clone it and run it.

```bash
./ms doctor                      # resolves config, lists callable models, smoke completion
./ms chat "explain rerank in one sentence"
./ms models --grep qwen3
```

Fourteen MCP tools over the whole surface — chat and vision, embeddings, rerank, image, video,
transcription, speech, batch, fine-tuning, files, async task polling, usage and budget.

---

## The budget guard

The part that matters if you are handing an agent an API key.

**The guard is on by default.** Per-model caps on tokens, images and TTS characters, checked
before the request leaves your machine:

```bash
./ms budget                      # free-quota headroom per model, and whether the guard is enforcing
MODEL_STUDIO_FREE_ONLY=0 ./ms …  # deliberately disable it
```

The suite asserts the property that actually matters: that **the guard, not the network, is what
stops a capped call**. A guard that only appears to work because the request failed anyway is not
a guard.

Every call is written to a local ledger, so `ms usage` tells you what you actually spent, per kind
and per model, without asking the console.

## Install

```bash
git clone https://github.com/shakilmahmud-posh/model-studio-mcp
cd model-studio-mcp
cp .model-studio.env.example ~/.model-studio.env
chmod 600 ~/.model-studio.env    # then put your key in it
./ms doctor
```

`MODEL_STUDIO_ENV` overrides the location entirely.

If your key arrived as a file from the console, `install-key.sh` will move it in for you:

```bash
bash install-key.sh ~/Downloads/<the-downloaded-file>
```

Console exports are frequently UTF-16 or GB18030 rather than UTF-8, so it tries several decodings
before giving up. **The key never reaches stdout, a log, or an agent session** — success prints
only the key's length and last four characters.

## Use it as an MCP server

```json
{
  "mcpServers": {
    "model-studio": {
      "command": "python3",
      "args": ["/absolute/path/to/model-studio-mcp/mcp_server.py"]
    }
  }
}
```

JSON-RPC 2.0 over newline-delimited stdio, hand-rolled — there is no MCP SDK dependency to keep in
sync. **stdout carries protocol frames only; every diagnostic goes to stderr**, which is the rule
most stdio servers break first and the suite checks explicitly.

Long jobs (video, ASR) default to submit-and-return, so a tool call never blocks an agent for five
minutes. Poll with `ms_task`.

| Tool                     |                                                                                |
| ------------------------ | ------------------------------------------------------------------------------ |
| `ms_doctor`              | credentials and endpoints end to end. Run this first when anything looks wrong |
| `ms_models`              | model ids this account can actually call                                       |
| `ms_chat`                | text or vision completion (Qwen, DeepSeek, GLM, Kimi)                          |
| `ms_embed`               | text-embedding-v4, auto-chunked past the 10-item request cap                   |
| `ms_rerank`              | qwen3-rerank, up to 500 documents                                              |
| `ms_image`               | text-to-image, or edit when reference images are supplied                      |
| `ms_video`               | text-to-video, or from a first-frame image                                     |
| `ms_transcribe`          | audio transcription from publicly reachable URLs                               |
| `ms_speak`               | speech synthesis, downloadable to a path                                       |
| `ms_task`                | poll any async task id — ids stay valid 24h                                    |
| `ms_files`               | upload, list, delete for fine-tuning and batch                                 |
| `ms_batch`               | batch inference, billed at 50% of real-time                                    |
| `ms_tune`                | fine-tuning lifecycle: create, status, logs, checkpoints                       |
| `ms_budget` · `ms_usage` | guard headroom, and the local ledger of what you spent                         |

## Prior art

[`nirholas/alibaba-cloud-mcp`](https://github.com/nirholas/alibaba-cloud-mcp) covers chat,
embeddings and model discovery. There is, as far as I can find, **no official MCP server from
Alibaba** for Model Studio.

This one covers the full surface, has no dependencies, and enforces a spend cap. If something
better exists, an issue pointing at it is welcome — I would rather cite it than pretend the space
is emptier than it is.

## One thing the docs will not tell you

**`/models` is not exhaustive.** `qwen3-rerank` works perfectly while being absent from the model
list. So nothing here gates on that list — it is offered as discovery, never as validation. If you
build on this API, do not reject a model id because the catalogue does not mention it.

## Verified

`bash verify.sh` — 49 static checks: credential resolution, key redaction, stdin handling, the
budget guard, and MCP protocol behaviour (initialize handshake, protocol-version echo, `tools/list`
surface, unknown tool returns an error rather than crashing, unconfigured call returns `isError`
rather than a stack trace, stdout hygiene).

Static mode makes no network calls and needs no credentials, so it runs on a fork's pull request
unchanged. CI additionally proves the stdlib-only claim by walking every import with `ast`, and
runs the suite on Python 3.9 through 3.13.

`bash verify.sh --deep` adds live API calls. **It spends real money**, which is why it is never
wired to CI.

## Security posture

The API key is the whole risk surface, so:

- it lives outside the repo, in a `chmod 600` file
- `config.describe()` never returns it — there is a test asserting that, and another asserting no
  realistic key shape ever appears in CLI output
- `install-key.sh` prints only length and last four characters
- the local ledger records what you called, never what you sent

## Maintenance

Best-effort, and I would rather say so than imply otherwise. Issues are welcome and I read them;
responses are not guaranteed and may be slow. Fork freely — that is what the licence is for.

## Licence

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by Alibaba Cloud.
