#!/usr/bin/env python3
"""MCP stdio server for Alibaba Cloud Model Studio.

Speaks JSON-RPC 2.0 over newline-delimited stdin/stdout — the MCP stdio
transport — with no SDK and no venv, so there is nothing to install or keep in
sync. stdout carries protocol frames ONLY; every diagnostic goes to stderr.

Long jobs (video, ASR) default to submit-and-return so a tool call never blocks
for minutes. Poll the returned task_id with the ms_task tool.
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelstudio import api, budget  # noqa: E402
from modelstudio.client import Client, download  # noqa: E402
from modelstudio.config import ConfigError, load  # noqa: E402

SERVER_INFO = {"name": "model-studio", "version": "1.0.0"}
FALLBACK_PROTOCOL = "2025-06-18"

_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = Client(load(require_key=True))
    return _client


def log(msg: str) -> None:
    print(f"[model-studio] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- schemas

def S(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
STRS = {"type": "array", "items": {"type": "string"}}

TOOLS: list[dict] = [
    {
        "name": "ms_doctor",
        "description": "Verify Model Studio credentials and endpoints end to end: resolve config, "
                       "list callable models, and run a smoke completion. Run this first when "
                       "anything looks wrong. Never reveals the API key.",
        "inputSchema": S({}),
    },
    {
        "name": "ms_models",
        "description": "List the model ids this account can actually call, optionally filtered. "
                       "Model availability differs per region and changes often, so prefer this "
                       "over assuming a model name exists.",
        "inputSchema": S({"grep": {**STR, "description": "case-insensitive substring filter"}}),
    },
    {
        "name": "ms_chat",
        "description": "Text or vision completion on a Qwen/DeepSeek/GLM model. Attach images by "
                       "URL or local path to use it as a vision model. Returns the text and the "
                       "token usage for the call.",
        "inputSchema": S({
            "prompt": STR,
            "model": {**STR, "description": "defaults to the configured default model"},
            "system": STR,
            "images": {**STRS, "description": "image URLs or local file paths"},
            "max_tokens": INT,
            "temperature": {"type": "number"},
            "thinking": {**BOOL, "description": "enable reasoning on hybrid models"},
        }, ["prompt"]),
    },
    {
        "name": "ms_embed",
        "description": "Embed texts with text-embedding-v4. Chunks past the 10-item per-request "
                       "cap automatically. Writes vectors to out_path when given, otherwise "
                       "returns only the count and dimension so a large payload never floods context.",
        "inputSchema": S({
            "texts": STRS,
            "model": STR,
            "dimensions": {**INT, "description": "2048|1536|1024|768|512|256|128|64"},
            "out_path": {**STR, "description": "write [{text,embedding}] JSON here"},
        }, ["texts"]),
    },
    {
        "name": "ms_rerank",
        "description": "Rerank candidate documents against a query with qwen3-rerank. "
                       "Max 500 documents; query plus documents must stay under 30k tokens.",
        "inputSchema": S({"query": STR, "documents": STRS, "model": STR, "top_n": INT},
                         ["query", "documents"]),
    },
    {
        "name": "ms_image",
        "description": "Generate an image from text, or edit images when reference images are "
                       "supplied. Downloads to out_dir when given; otherwise returns the URLs.",
        "inputSchema": S({
            "prompt": STR,
            "model": {**STR, "description": "e.g. qwen-image-3.0-pro; wan* models run async"},
            "images": {**STRS, "description": "reference image URLs for edit mode"},
            "size": {**STR, "description": "e.g. 1024*1024"},
            "n": INT,
            "negative": STR,
            "out_dir": STR,
        }, ["prompt"]),
    },
    {
        "name": "ms_video",
        "description": "Generate video from text, or from a first-frame image URL. Long-running "
                       "(1-5 min): by default this submits and returns a task_id to poll with "
                       "ms_task. Set wait=true only when blocking is acceptable.",
        "inputSchema": S({
            "prompt": STR,
            "model": STR,
            "image": {**STR, "description": "public first-frame image URL for image-to-video"},
            "resolution": STR,
            "duration": INT,
            "wait": {**BOOL, "description": "block until done (default false)"},
            "out_path": STR,
        }, ["prompt"]),
    },
    {
        "name": "ms_transcribe",
        "description": "Transcribe audio files. Takes PUBLICLY REACHABLE URLs only — there is no "
                       "local upload path, so host the file first. Async: returns a task_id "
                       "unless wait=true.",
        "inputSchema": S({
            "urls": STRS,
            "model": STR,
            "language": {**STR, "description": "hint, e.g. bn / en / ms"},
            "wait": BOOL,
        }, ["urls"]),
    },
    {
        "name": "ms_speak",
        "description": "Synthesize speech from text. Downloads to out_path when given, "
                       "otherwise returns the audio URL.",
        "inputSchema": S({"text": STR, "model": STR, "voice": STR, "language": STR, "out_path": STR},
                         ["text"]),
    },
    {
        "name": "ms_task",
        "description": "Poll any async task id (image, video, ASR). Task ids stay valid for 24h. "
                       "Status flows PENDING -> RUNNING -> SUCCEEDED | FAILED.",
        "inputSchema": S({"task_id": STR, "wait": BOOL}, ["task_id"]),
    },
    {
        "name": "ms_files",
        "description": "Manage files on Model Studio: upload (for fine-tuning or batch), list, "
                       "delete, or read back contents.",
        "inputSchema": S({
            "action": {"type": "string", "enum": ["upload", "list", "delete", "cat"]},
            "path": {**STR, "description": "local path for upload"},
            "file_id": STR,
            "purpose": {**STR, "description": "fine-tune | batch"},
        }, ["action"]),
    },
    {
        "name": "ms_batch",
        "description": "Batch inference, billed at 50 percent of real-time for the same model. "
                       "Create from a local JSONL path or an already-uploaded file id.",
        "inputSchema": S({
            "action": {"type": "string", "enum": ["create", "get", "list", "cancel"]},
            "target": {**STR, "description": "JSONL path or file id (create), else batch id"},
            "endpoint": STR,
        }, ["action"]),
    },
    {
        "name": "ms_tune",
        "description": "Fine-tuning lifecycle: create a job, list, check status, read logs, list "
                       "checkpoints, deploy a checkpoint as a callable model, cancel, delete, or "
                       "run a deployed model. One job trains at a time; the rest sit in QUEUING.",
        "inputSchema": S({
            "action": {"type": "string",
                       "enum": ["create", "list", "get", "logs", "checkpoints", "deploy",
                                "cancel", "delete", "run"]},
            "job_id": STR,
            "model": {**STR, "description": "base model for create, or deployed model id for run"},
            "training_file_ids": STRS,
            "training_type": {"type": "string",
                              "enum": ["sft", "efficient_sft", "cpt", "dpo_full", "dpo_lora"]},
            "hyper_parameters": {"type": "object"},
            "checkpoint_id": STR,
            "model_name": STR,
            "prompt": STR,
        }, ["action"]),
    },
    {
        "name": "ms_budget",
        "description": "Free-quota headroom per model, and whether the local guard is enforcing. "
                       "Counts only calls made through this tooling — the real protection against "
                       "charges is the per-model 'Free Quota Only' switch in the Model Studio console.",
        "inputSchema": S({}),
    },
    {
        "name": "ms_usage",
        "description": "Local ledger of every call made through this tool, aggregated per "
                       "kind/model. Free quota is per-model and expires 90 days after activation, "
                       "so this is how you see the burn without opening the console.",
        "inputSchema": S({"since": {**STR, "description": "ISO date, e.g. 2026-08-01"}}),
    },
]


# ---------------------------------------------------------------- handlers

def h_doctor(_):
    c = client()
    d = c.cfg.describe()
    models = api.list_models(c)
    probe = c.cfg.default_model if c.cfg.default_model in models else (models[0] if models else None)
    result = {"config": d, "model_count": len(models), "models": models}
    if c.cfg.region != "ap-southeast-1":
        result["warning"] = (f"region is {c.cfg.region}, not ap-southeast-1; "
                             "the International free quota exists only in Singapore")
    if probe:
        r = api.chat(c, "Reply with exactly: OK", model=probe, max_tokens=16,
                     temperature=0.0, stream=False, out=io.StringIO())
        result["smoke"] = {"model": probe, "text": r["text"], "usage": r["usage"],
                           "pass": "OK" in (r["text"] or "")}
    return result


def h_models(a):
    models = api.list_models(client())
    g = (a.get("grep") or "").lower()
    return [m for m in models if g in m.lower()] if g else models


def h_chat(a):
    r = api.chat(client(), a["prompt"], model=a.get("model"), system=a.get("system"),
                 images=a.get("images"), max_tokens=a.get("max_tokens", 2048),
                 temperature=a.get("temperature", 0.4), enable_thinking=a.get("thinking"),
                 stream=False, out=io.StringIO())
    return {"text": r["text"], "usage": r["usage"]}


def h_embed(a):
    c = client()
    texts = a["texts"]
    r = api.embed(c, texts, model=a.get("model", "text-embedding-v4"), dimensions=a.get("dimensions"))
    if a.get("out_path"):
        Path(a["out_path"]).write_text(json.dumps(
            [{"text": t, "embedding": v} for t, v in zip(texts, r["vectors"])]), encoding="utf-8")
        return {"count": len(r["vectors"]), "dim": r["dim"],
                "total_tokens": r["total_tokens"], "written_to": a["out_path"]}
    # Vectors are enormous; never return them inline by default.
    return {"count": len(r["vectors"]), "dim": r["dim"], "total_tokens": r["total_tokens"],
            "note": "pass out_path to persist the vectors"}


def h_rerank(a):
    return api.rerank(client(), a["query"], a["documents"],
                      model=a.get("model", "qwen3-rerank"), top_n=a.get("top_n", 10))


def h_image(a):
    c = client()
    r = api.image_generate(c, a["prompt"], model=a.get("model", "qwen-image-3.0-pro"),
                           images=a.get("images"), size=a.get("size"), n=a.get("n", 1),
                           negative=a.get("negative"))
    if a.get("out_dir") and r["urls"]:
        paths = [str(download(u, Path(a["out_dir"]) / f"image-{i}.png"))
                 for i, u in enumerate(r["urls"])]
        return {"paths": paths, "urls": r["urls"]}
    return {"urls": r["urls"]}


def h_video(a):
    c = client()
    r = api.video_generate(c, a["prompt"], model=a.get("model", "wan2.6-i2v-flash"),
                           image=a.get("image"), resolution=a.get("resolution", "720P"),
                           duration=a.get("duration"), wait=bool(a.get("wait", False)))
    if r.get("url") and a.get("out_path"):
        r["path"] = str(download(r["url"], Path(a["out_path"])))
    if not r.get("url"):
        r["next"] = f"poll with ms_task task_id={r['task_id']}"
    return r


def h_transcribe(a):
    return api.transcribe(client(), a["urls"], model=a.get("model", "qwen3-asr-flash-filetrans"),
                          language=a.get("language"), wait=bool(a.get("wait", False)))


def h_speak(a):
    c = client()
    r = api.speak(c, a["text"], model=a.get("model", "qwen3-tts-flash"),
                  voice=a.get("voice", "Cherry"), language=a.get("language"))
    if r.get("url") and a.get("out_path"):
        r["path"] = str(download(r["url"], Path(a["out_path"])))
    return r


def h_task(a):
    c = client()
    return c.wait_task(a["task_id"]) if a.get("wait") else c.get_task(a["task_id"])


def h_files(a):
    c, action = client(), a["action"]
    if action == "upload":
        return api.file_upload(c, Path(a["path"]), a.get("purpose", "fine-tune"))
    if action == "list":
        return api.file_list(c)
    if action == "delete":
        return api.file_delete(c, a["file_id"])
    return {"content": api.file_content(c, a["file_id"])}


def h_batch(a):
    c, action = client(), a["action"]
    if action == "create":
        t = a["target"]
        fid = api.file_upload(c, Path(t), "batch")["id"] if Path(t).exists() else t
        return api.batch_create(c, fid, endpoint=a.get("endpoint", "/v1/chat/completions"))
    if action == "get":
        return api.batch_get(c, a["target"])
    if action == "list":
        return api.batch_list(c)
    return api.batch_cancel(c, a["target"])


def h_tune(a):
    c, action = client(), a["action"]
    if action == "create":
        return api.tune_create(c, a["model"], a["training_file_ids"],
                               training_type=a.get("training_type", "efficient_sft"),
                               hyper=a.get("hyper_parameters"))
    if action == "list":
        return api.tune_list(c)
    if action == "get":
        return api.tune_get(c, a["job_id"])
    if action == "logs":
        return api.tune_logs(c, a["job_id"])
    if action == "checkpoints":
        return api.tune_checkpoints(c, a["job_id"])
    if action == "deploy":
        return api.tune_deploy(c, a["job_id"], a["checkpoint_id"], a["model_name"])
    if action == "cancel":
        return api.tune_cancel(c, a["job_id"])
    if action == "delete":
        return api.tune_delete(c, a["job_id"])
    return api.tuned_generate(c, a["model"], a["prompt"])


def h_budget(a):
    c = client()
    return {"enforcing": budget.enabled(), "limits": budget.limits(),
            "models": budget.report(c.cfg.ledger_path),
            "note": "local seatbelt only; the console 'Free Quota Only' switch is the wall"}


def h_usage(a):
    from collections import defaultdict
    c = client()
    p = Path(c.cfg.ledger_path)
    if not p.exists():
        return {"ledger": str(p), "rows": 0, "note": "no calls recorded yet"}
    agg = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "total": 0})
    since = a.get("since") or ""
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since and row.get("ts", "") < since:
            continue
        k = f"{row.get('kind')}/{row.get('model')}"
        agg[k]["calls"] += 1
        for src, dst in (("input_tokens", "in"), ("output_tokens", "out"), ("total_tokens", "total")):
            agg[k][dst] += row.get(src) or 0
    return {"ledger": str(p), "by_model": dict(agg)}


HANDLERS = {
    "ms_doctor": h_doctor, "ms_models": h_models, "ms_chat": h_chat, "ms_embed": h_embed,
    "ms_rerank": h_rerank, "ms_image": h_image, "ms_video": h_video,
    "ms_transcribe": h_transcribe, "ms_speak": h_speak, "ms_task": h_task,
    "ms_files": h_files, "ms_batch": h_batch, "ms_tune": h_tune, "ms_usage": h_usage,
    "ms_budget": h_budget,
}


# ---------------------------------------------------------------- protocol

def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def reply(rid, result: dict) -> None:
    send({"jsonrpc": "2.0", "id": rid, "result": result})


def error(rid, code: int, message: str) -> None:
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def handle(msg: dict) -> None:
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}

    if method == "initialize":
        # Echo the client's protocol version when it sends one; interop beats pinning.
        version = params.get("protocolVersion") or FALLBACK_PROTOCOL
        reply(rid, {"protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO})
        return

    if method in ("notifications/initialized", "notifications/cancelled"):
        return  # notifications carry no id and take no response

    if method == "ping":
        reply(rid, {})
        return

    if method == "tools/list":
        reply(rid, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = HANDLERS.get(name)
        if not fn:
            error(rid, -32601, f"unknown tool: {name}")
            return
        try:
            result = fn(args)
            text = result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False)
            reply(rid, {"content": [{"type": "text", "text": text}]})
        except ConfigError as e:
            reply(rid, {"content": [{"type": "text", "text": f"not configured: {e}"}], "isError": True})
        except Exception as e:  # surface the cause to the caller, keep the server alive
            log(traceback.format_exc())
            reply(rid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                        "isError": True})
        return

    if rid is not None:
        error(rid, -32601, f"unknown method: {method}")


def main() -> int:
    log(f"ready — {len(TOOLS)} tools on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log(f"dropped non-JSON line ({len(line)} bytes)")
            continue
        try:
            handle(msg)
        except Exception:
            log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
