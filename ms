#!/usr/bin/env python3
"""ms — Alibaba Cloud Model Studio from the command line.

    ms doctor                              prove the credential end to end
    ms models [--grep TEXT]                what this account can actually call
    ms chat "..." [-f FILE] [--image X]    text or vision
    ms embed "a" "b" [--dim 1024]          embeddings (chunks past the 10-item cap)
    ms rerank --query Q --docs docs.txt    rerank candidates
    ms image "..." [--out DIR]             text-to-image / image edit
    ms video "..." [--image URL]           text-to-video / image-to-video
    ms asr URL [URL...]                    async file transcription
    ms tts "..." [--out speech.wav]        speech synthesis
    ms task TASK_ID                        poll or resume any async task
    ms files upload|list|delete|cat
    ms batch create|get|list|cancel        50% of real-time price
    ms tune create|list|get|logs|checkpoints|deploy|cancel|delete|run
    ms usage [--days N]                    local ledger: what you have burned
    ms anthropic                           env exports to point Claude Code at Qwen

Add --json to almost anything for machine-readable output.
Credentials live in ~/dev/.model-studio.env and are never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelstudio import api, budget  # noqa: E402
from modelstudio.client import Client, download, read_stdin  # noqa: E402
from modelstudio.config import ConfigError, load  # noqa: E402


def out_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def progress(status: str, waited: int) -> None:
    print(f"  [{waited:>4}s] {status}", file=sys.stderr)


def read_lines(spec: list[str], file: Path | None) -> list[str]:
    """Items come from args, or one-per-line from a file, or stdin."""
    if file:
        return [l for l in file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if spec:
        return spec
    return [l for l in read_stdin().splitlines() if l.strip()]


# ---------------------------------------------------------------- commands

def cmd_doctor(c: Client, a) -> int:
    d = c.cfg.describe()
    print("config")
    for k, v in d.items():
        print(f"  {k:<16} {v}")

    if c.cfg.region != "ap-southeast-1":
        print(f"\n  ! region is {c.cfg.region}, not ap-southeast-1. The International "
              f"free quota exists only in Singapore.", file=sys.stderr)

    print("\nreachability")
    models = api.list_models(c)
    print(f"  /models          {len(models)} models visible")
    if not models:
        print("  ! no models returned — activation or region is wrong", file=sys.stderr)
        return 1

    probe = c.cfg.default_model if c.cfg.default_model in models else models[0]
    print(f"  probe model      {probe}")
    r = api.chat(c, "Reply with exactly: OK", model=probe, max_tokens=16,
                 temperature=0.0, stream=False, out=sys.stdout)
    ok = "OK" in (r["text"] or "")
    print(f"  /chat            {'pass' if ok else 'FAIL'} — {r['text'][:60]!r}")
    print(f"  tokens           {r['usage']}")

    print("\nmodels")
    for m in models:
        print(f"  {m}")
    print(f"\nledger           {c.cfg.ledger_path}")
    return 0 if ok else 1


def cmd_models(c: Client, a) -> int:
    models = api.list_models(c)
    if a.grep:
        models = [m for m in models if a.grep.lower() in m.lower()]
    out_json(models) if a.json else print("\n".join(models))
    return 0


def cmd_chat(c: Client, a) -> int:
    prompt = " ".join(a.prompt)
    stdin_text = read_stdin()
    if a.file:
        prompt += f"\n\n=== file: {a.file} ===\n{Path(a.file).read_text(encoding='utf-8', errors='replace')}"
    if stdin_text.strip():
        prompt = f"{prompt}\n\n=== stdin ===\n{stdin_text}" if prompt.strip() else stdin_text
    if not prompt.strip():
        print("error: no prompt (pass args, pipe stdin, or use -f)", file=sys.stderr)
        return 2

    r = api.chat(c, prompt, model=a.model, system=a.system, images=a.image, audio=a.audio,
                 max_tokens=a.max_tokens, temperature=a.temperature,
                 stream=not (a.no_stream or a.json), timeout=a.timeout,
                 enable_thinking=a.thinking if a.thinking is not None else None,
                 out=sys.stderr if a.json else sys.stdout)
    if a.json:
        out_json({"text": r["text"], "usage": r["usage"]})
    elif r.get("usage") and not a.quiet:
        u = r["usage"]
        print(f"[{a.model or c.cfg.default_model}] in={u.get('prompt_tokens')} "
              f"out={u.get('completion_tokens')} total={u.get('total_tokens')}", file=sys.stderr)
    return 0


def cmd_embed(c: Client, a) -> int:
    texts = read_lines(a.text, a.file)
    if not texts:
        print("error: no input text", file=sys.stderr)
        return 2
    r = api.embed(c, texts, model=a.model, dimensions=a.dim)
    if a.out:
        Path(a.out).write_text(json.dumps(
            [{"text": t, "embedding": v} for t, v in zip(texts, r["vectors"])]), encoding="utf-8")
        print(f"{len(r['vectors'])} vectors (dim {r['dim']}, {r['total_tokens']} tokens) -> {a.out}")
    elif a.json:
        out_json(r)
    else:
        print(f"{len(r['vectors'])} vectors, dim {r['dim']}, {r['total_tokens']} tokens")
    return 0


def cmd_rerank(c: Client, a) -> int:
    docs = read_lines(a.doc, a.docs)
    if not docs:
        print("error: no documents", file=sys.stderr)
        return 2
    r = api.rerank(c, a.query, docs, model=a.model, top_n=a.top_n)
    if a.json:
        out_json(r)
    else:
        for hit in r.get("results", []):
            idx, score = hit.get("index"), hit.get("relevance_score")
            text = (hit.get("document") or {}).get("text") or (docs[idx] if idx is not None else "")
            print(f"{score:.4f}  {text[:110]}")
    return 0


def cmd_image(c: Client, a) -> int:
    r = api.image_generate(c, " ".join(a.prompt), model=a.model, images=a.image,
                           size=a.size, n=a.n, negative=a.negative, path=a.path,
                           on_poll=progress)
    if a.json:
        out_json(r)
        return 0
    if not r["urls"]:
        print("no image url in response; re-run with --json to inspect", file=sys.stderr)
        return 1
    for i, url in enumerate(r["urls"]):
        if a.out:
            dest = Path(a.out) / f"image-{i}.png"
            download(url, dest)
            print(dest)
        else:
            print(url)
    return 0


def cmd_video(c: Client, a) -> int:
    r = api.video_generate(c, " ".join(a.prompt), model=a.model, image=a.image, audio=a.audio,
                           resolution=a.resolution, duration=a.duration,
                           watermark=a.watermark, path=a.path, on_poll=progress)
    if a.json:
        out_json(r)
        return 0
    if not r["url"]:
        print(f"no video url; poll again with: ms task {r['task_id']}", file=sys.stderr)
        return 1
    print(download(r["url"], Path(a.out)) if a.out else r["url"])
    return 0


def cmd_asr(c: Client, a) -> int:
    r = api.transcribe(c, a.url, model=a.model, language=a.language, path=a.path, on_poll=progress)
    if a.json:
        out_json(r)
        return 0
    for item in r["results"]:
        url = item.get("transcription_url")
        print(f"{item.get('file_url', '?')} -> {url or item.get('subtask_status')}")
    return 0


def cmd_tts(c: Client, a) -> int:
    r = api.speak(c, " ".join(a.text), model=a.model, voice=a.voice, language=a.language, path=a.path)
    if a.json:
        out_json(r)
        return 0
    if not r["url"]:
        print("no audio url; re-run with --json to inspect", file=sys.stderr)
        return 1
    print(download(r["url"], Path(a.out)) if a.out else r["url"])
    return 0


def cmd_task(c: Client, a) -> int:
    if a.wait:
        out_json(c.wait_task(a.task_id, on_poll=progress))
    else:
        out_json(c.get_task(a.task_id))
    return 0


def cmd_files(c: Client, a) -> int:
    if a.action == "upload":
        out_json(api.file_upload(c, Path(a.target), a.purpose))
    elif a.action == "list":
        rows = api.file_list(c)
        if a.json:
            out_json(rows)
        else:
            for f in rows:
                print(f"{f.get('id'):<28} {f.get('bytes', 0):>10}  {f.get('filename')}")
    elif a.action == "delete":
        out_json(api.file_delete(c, a.target))
    elif a.action == "cat":
        print(api.file_content(c, a.target))
    return 0


def cmd_batch(c: Client, a) -> int:
    if a.action == "create":
        f = api.file_upload(c, Path(a.target), "batch") if Path(a.target).exists() else {"id": a.target}
        job = api.batch_create(c, f["id"], endpoint=a.endpoint, window=a.window)
        out_json(job)
    elif a.action == "get":
        out_json(api.batch_get(c, a.target))
    elif a.action == "list":
        out_json(api.batch_list(c))
    elif a.action == "cancel":
        out_json(api.batch_cancel(c, a.target))
    return 0


def cmd_tune(c: Client, a) -> int:
    if a.action == "create":
        hyper = json.loads(a.hyper) if a.hyper else None
        out_json(api.tune_create(c, a.model, a.file, training_type=a.type, hyper=hyper,
                                 validation_file_ids=a.validation))
    elif a.action == "list":
        out_json(api.tune_list(c))
    elif a.action == "get":
        out_json(api.tune_get(c, a.target))
    elif a.action == "logs":
        out_json(api.tune_logs(c, a.target))
    elif a.action == "checkpoints":
        out_json(api.tune_checkpoints(c, a.target))
    elif a.action == "deploy":
        out_json(api.tune_deploy(c, a.target, a.checkpoint, a.name))
    elif a.action == "cancel":
        out_json(api.tune_cancel(c, a.target))
    elif a.action == "delete":
        out_json(api.tune_delete(c, a.target))
    elif a.action == "run":
        out_json(api.tuned_generate(c, a.target, " ".join(a.prompt)))
    return 0


def cmd_usage(c: Client, a) -> int:
    p = Path(c.cfg.ledger_path)
    if not p.exists():
        print(f"no ledger yet at {p}")
        return 0
    # Media services do not bill in tokens, so track their units too — otherwise
    # image/TTS/video rows read as free.
    UNITS = (("characters", "chars"), ("output_image_count", "images"),
             ("duration", "sec"), ("video_duration", "sec"))
    agg = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "total": 0, "other": defaultdict(int)})
    cutoff = a.since or ""
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff and row.get("ts", "") < cutoff:
            continue
        k = f"{row.get('kind')}/{row.get('model')}"
        agg[k]["calls"] += 1
        for src, dst in (("input_tokens", "in"), ("output_tokens", "out"), ("total_tokens", "total")):
            agg[k][dst] += row.get(src) or 0
        for src, label in UNITS:
            if row.get(src):
                agg[k]["other"][label] += row[src]
    if a.json:
        out_json({k: {**v, "other": dict(v["other"])} for k, v in agg.items()})
        return 0
    print(f"{'kind/model':<40} {'calls':>6} {'in':>9} {'out':>9} {'total':>9}  other")
    for k in sorted(agg, key=lambda k: -agg[k]["total"]):
        v = agg[k]
        other = " ".join(f"{n} {lbl}" for lbl, n in sorted(v["other"].items())) or "-"
        print(f"{k:<40} {v['calls']:>6} {v['in']:>9} {v['out']:>9} {v['total']:>9}  {other}")
    print(f"\nfree quota is per-model (~1M tokens each) and expires 90 days after activation.")
    return 0


def cmd_budget(c: Client, a) -> int:
    rows = budget.report(c.cfg.ledger_path)
    if a.json:
        out_json({"enforcing": budget.enabled(), "limits": budget.limits(), "models": rows})
        return 0
    print(f"local guard : {'ON — calls blocked at the cap' if budget.enabled() else 'OFF — paid spend allowed'}")
    print(f"ledger      : {c.cfg.ledger_path}")
    print(f"caps        : {budget.limits()}\n")
    if not rows:
        print("nothing spent yet")
    for r in rows:
        model = r.pop("model")
        parts = " ".join(f"{k}={v['used']:,}/{v['cap']:,} ({v['pct']}%)" for k, v in r.items())
        print(f"  {model:<34} {parts}")
    print("\nThis counts ONLY calls made through this tooling. The real protection is")
    print("the per-model 'Free Quota Only' switch in the Model Studio console.")
    return 0


def cmd_anthropic(c: Client, a) -> int:
    """Model Studio speaks the Anthropic Messages protocol too, so an Anthropic
    client can be pointed at Qwen by swapping two env vars."""
    print(f"export ANTHROPIC_BASE_URL={c.cfg.anthropic_base}")
    print(f"export ANTHROPIC_AUTH_TOKEN=$(grep '^DASHSCOPE_API_KEY=' ~/dev/.model-studio.env | cut -d= -f2-)")
    print(f"# then pick a model with --model, e.g. {c.cfg.default_model}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")

    p = argparse.ArgumentParser(prog="ms", description="Alibaba Cloud Model Studio CLI",
                                epilog="--json works on any subcommand, before or after it")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="prove credentials and endpoints end to end", parents=[common]).set_defaults(fn=cmd_doctor)

    m = sub.add_parser("models", help="list callable model ids", parents=[common])
    m.add_argument("--grep", help="substring filter")
    m.set_defaults(fn=cmd_models)

    ch = sub.add_parser("chat", help="text or vision completion", parents=[common])
    ch.add_argument("prompt", nargs="*")
    ch.add_argument("-f", "--file", help="attach a file as context")
    ch.add_argument("--model")
    ch.add_argument("--system")
    ch.add_argument("--image", action="append", help="url or local path (repeatable)")
    ch.add_argument("--audio", action="append")
    ch.add_argument("--max-tokens", type=int, default=2048)
    ch.add_argument("--temperature", type=float, default=0.4)
    ch.add_argument("--thinking", type=lambda s: s.lower() == "true", default=None,
                    help="true/false for hybrid reasoning models")
    ch.add_argument("--no-stream", action="store_true")
    ch.add_argument("--timeout", type=int, default=300,
                    help="seconds before giving up (default 300)")
    ch.add_argument("--quiet", action="store_true", help="suppress the token-usage line")
    ch.set_defaults(fn=cmd_chat)

    e = sub.add_parser("embed", help="text embeddings", parents=[common])
    e.add_argument("text", nargs="*")
    e.add_argument("--file", type=Path, help="one text per line")
    e.add_argument("--model", default="text-embedding-v4")
    e.add_argument("--dim", type=int)
    e.add_argument("--out", help="write [{text,embedding}] JSON here")
    e.set_defaults(fn=cmd_embed)

    r = sub.add_parser("rerank", help="rerank candidate documents", parents=[common])
    r.add_argument("doc", nargs="*")
    r.add_argument("--query", required=True)
    r.add_argument("--docs", type=Path, help="one document per line")
    r.add_argument("--model", default="qwen3-rerank")
    r.add_argument("--top-n", type=int, default=10)
    r.set_defaults(fn=cmd_rerank)

    i = sub.add_parser("image", help="text-to-image, or edit with --image", parents=[common])
    i.add_argument("prompt", nargs="+")
    i.add_argument("--model", default="qwen-image-3.0-pro")
    i.add_argument("--image", action="append", help="reference image url (edit mode)")
    i.add_argument("--size", help="e.g. 1024*1024")
    i.add_argument("-n", type=int, default=1)
    i.add_argument("--negative")
    i.add_argument("--out", help="download into this directory")
    i.add_argument("--path", help="override the API route")
    i.set_defaults(fn=cmd_image)

    v = sub.add_parser("video", help="text-to-video, or image-to-video with --image", parents=[common])
    v.add_argument("prompt", nargs="+")
    v.add_argument("--model", default="wan2.6-i2v-flash")
    v.add_argument("--image", help="first-frame image URL (public)")
    v.add_argument("--audio", help="audio URL for lip-sync models")
    v.add_argument("--resolution", default="720P")
    v.add_argument("--duration", type=int)
    v.add_argument("--watermark", action="store_true")
    v.add_argument("--out", help="download to this path")
    v.add_argument("--path", help="override the API route")
    v.set_defaults(fn=cmd_video)

    s = sub.add_parser("asr", help="async transcription of public audio URLs", parents=[common])
    s.add_argument("url", nargs="+")
    s.add_argument("--model", default="qwen3-asr-flash-filetrans")
    s.add_argument("--language", help="hint, e.g. bn / en / ms")
    s.add_argument("--path", help="override the API route")
    s.set_defaults(fn=cmd_asr)

    t = sub.add_parser("tts", help="speech synthesis", parents=[common])
    t.add_argument("text", nargs="+")
    t.add_argument("--model", default="qwen3-tts-flash")
    t.add_argument("--voice", default="Cherry")
    t.add_argument("--language")
    t.add_argument("--out", help="download to this path")
    t.add_argument("--path", help="override the API route")
    t.set_defaults(fn=cmd_tts)

    tk = sub.add_parser("task", help="poll or resume any async task id", parents=[common])
    tk.add_argument("task_id")
    tk.add_argument("--wait", action="store_true", help="block until terminal")
    tk.set_defaults(fn=cmd_task)

    f = sub.add_parser("files", help="upload / list / delete / cat files", parents=[common])
    f.add_argument("action", choices=["upload", "list", "delete", "cat"])
    f.add_argument("target", nargs="?")
    f.add_argument("--purpose", default="fine-tune")
    f.set_defaults(fn=cmd_files)

    b = sub.add_parser("batch", help="batch inference at 50%% of real-time price", parents=[common])
    b.add_argument("action", choices=["create", "get", "list", "cancel"])
    b.add_argument("target", nargs="?", help="jsonl path or file id (create) / batch id")
    b.add_argument("--endpoint", default="/v1/chat/completions")
    b.add_argument("--window", default="24h")
    b.set_defaults(fn=cmd_batch)

    tn = sub.add_parser("tune", help="fine-tuning jobs", parents=[common])
    tn.add_argument("action", choices=["create", "list", "get", "logs", "checkpoints",
                                       "deploy", "cancel", "delete", "run"])
    tn.add_argument("target", nargs="?", help="job id, or deployed model id for `run`")
    tn.add_argument("prompt", nargs="*", help="prompt for `run`")
    tn.add_argument("--model", help="base model for `create`")
    tn.add_argument("--file", action="append", default=[], help="training file id (repeatable)")
    tn.add_argument("--validation", action="append", default=[])
    tn.add_argument("--type", default="efficient_sft",
                    choices=["sft", "efficient_sft", "cpt", "dpo_full", "dpo_lora"])
    tn.add_argument("--hyper", help="JSON hyper_parameters override")
    tn.add_argument("--checkpoint", help="checkpoint id for `deploy`")
    tn.add_argument("--name", help="published model name for `deploy`")
    tn.set_defaults(fn=cmd_tune)

    u = sub.add_parser("usage", help="local ledger of what you have spent", parents=[common])
    u.add_argument("--since", help="ISO date, e.g. 2026-08-01")
    u.set_defaults(fn=cmd_usage)

    sub.add_parser("budget", help="free-quota headroom per model", parents=[common]).set_defaults(fn=cmd_budget)

    sub.add_parser("anthropic", help="env exports to point an Anthropic client at Qwen", parents=[common]).set_defaults(fn=cmd_anthropic)
    return p


def hoist_global_flags(argv: list[str]) -> list[str]:
    """Accept `ms --json usage` as well as `ms usage --json`.

    argparse merges the subparser namespace over the parent, so a --json seen
    before the subcommand gets clobbered. Move it past the subcommand instead
    of fighting that. Only the leading option block is rewritten, so a --json
    inside a prompt is left alone.
    """
    lead = 0
    while lead < len(argv) and argv[lead].startswith("-"):
        lead += 1
    if "--json" not in argv[:lead]:
        return argv
    return [x for x in argv[:lead] if x != "--json"] + argv[lead:] + ["--json"]


def main() -> int:
    a = build_parser().parse_args(hoist_global_flags(sys.argv[1:]))
    try:
        cfg = load(require_key=True)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        return a.fn(Client(cfg), a)
    except (RuntimeError, ValueError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
