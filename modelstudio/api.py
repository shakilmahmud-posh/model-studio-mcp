"""Function surface for Model Studio.

Route map. `openai` = /compatible-mode/v1, `native` = /api/v1.

    chat, vision      openai   /chat/completions
    embed             openai   /embeddings
    rerank            native   /services/rerank/text-rerank/text-rerank
    image_generate    native   multimodal-generation | text2image (async)
    video_generate    native   /services/aigc/video-generation/video-synthesis (async)
    transcribe        native   /services/audio/asr/transcription (async)
    speak             native   /services/aigc/multimodal-generation/generation
    files, batches    openai   /files, /batches
    tuning            native   /fine-tunes

CONFIDENCE. chat / vision / embed / models / files / batches / fine-tunes are
transcribed from official API references. The image, video, ASR and TTS request
shapes come from per-model reference pages and vary by model family, so every one
of them takes a `path` override and passes `extra` through to the body. When a
model 404s or 400s, that is the knob to reach for — not a rewrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .client import Client, data_uri

# --------------------------------------------------------------------------
# account
# --------------------------------------------------------------------------

def list_models(c: Client) -> list[str]:
    payload = c.get(f"{c.cfg.openai_base}/models", timeout=60)
    return sorted(m.get("id", "") for m in payload.get("data", []) if m.get("id"))


# --------------------------------------------------------------------------
# text + vision
# --------------------------------------------------------------------------

def _content(prompt: str, images: list[str] | None, audio: list[str] | None):
    """Plain string for text-only, OpenAI content-parts list when media is attached."""
    if not images and not audio:
        return prompt
    parts: list[dict] = []
    for src in images or []:
        url = data_uri(Path(src)) if not src.startswith(("http://", "https://", "data:")) else src
        parts.append({"type": "image_url", "image_url": {"url": url}})
    for src in audio or []:
        parts.append({"type": "input_audio", "input_audio": {"data": src, "format": Path(src).suffix.lstrip(".")}})
    parts.append({"type": "text", "text": prompt})
    return parts


def chat(
    c: Client,
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    stream: bool = True,
    enable_thinking: bool | None = None,
    tools: list | None = None,
    extra: dict | None = None,
    timeout: int = 300,
    out=sys.stdout,
) -> dict:
    """One chat/vision turn. Returns {'text', 'usage', 'raw'}."""
    model = model or c.cfg.default_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _content(prompt, images, audio)})

    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": stream}
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if tools:
        body["tools"] = tools
    if stream:
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)

    url = f"{c.cfg.openai_base}/chat/completions"

    if not stream:
        payload = c.post(url, body, timeout=timeout)
        text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
        c.record("chat", model, payload.get("usage"))
        if text:
            out.write(text if text.endswith("\n") else text + "\n")
            out.flush()
        return {"text": text, "usage": payload.get("usage"), "raw": payload}

    resp = c.request("POST", url, body=body, raw=True, timeout=timeout)
    parts, usage, tool_calls = [], None, []
    with resp:
        for line in resp:
            s = line.decode("utf-8", errors="replace").strip()
            if not s.startswith("data:"):
                continue
            data = s[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                piece = delta.get("content") or ""
                if piece:
                    parts.append(piece)
                    out.write(piece)
                    out.flush()
    if parts:
        out.write("\n")
    c.record("chat", model, usage)
    return {"text": "".join(parts), "usage": usage, "tool_calls": tool_calls, "raw": None}


# --------------------------------------------------------------------------
# embeddings + rerank
# --------------------------------------------------------------------------

def embed(c: Client, texts: list[str], *, model: str = "text-embedding-v4",
          dimensions: int | None = None) -> dict:
    """Batch cap is 10 items per request; this chunks transparently."""
    vectors, total = [], 0
    for i in range(0, len(texts), 10):
        body = {"model": model, "input": texts[i:i + 10], "encoding_format": "float"}
        if dimensions:
            body["dimensions"] = dimensions
        payload = c.post(f"{c.cfg.openai_base}/embeddings", body)
        vectors.extend(d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"]))
        total += (payload.get("usage") or {}).get("total_tokens", 0)
    c.record("embed", model, {"total_tokens": total}, {"items": len(texts)})
    return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0, "total_tokens": total}


def rerank(c: Client, query: str, documents: list[str], *, model: str = "qwen3-rerank",
           top_n: int = 10, return_documents: bool = True) -> dict:
    """Max 500 documents, and query + docs together must stay under 30k tokens."""
    if len(documents) > 500:
        raise ValueError(f"rerank takes at most 500 documents, got {len(documents)}")
    body = {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": return_documents},
    }
    payload = c.post(f"{c.cfg.dashscope_base}/services/rerank/text-rerank/text-rerank", body)
    c.record("rerank", model, payload.get("usage"), {"docs": len(documents)})
    return (payload.get("output") or {})


# --------------------------------------------------------------------------
# image
# --------------------------------------------------------------------------

def image_generate(c: Client, prompt: str, *, model: str = "qwen-image-3.0-pro",
                   images: list[str] | None = None, size: str | None = None, n: int = 1,
                   negative: str | None = None, path: str | None = None,
                   extra: dict | None = None, on_poll=None) -> dict:
    """Text-to-image, or image edit when `images` is supplied.

    Two families with two shapes: qwen-image* answers synchronously on the
    multimodal-generation route, wan* runs async on the text2image route.
    `path` overrides the route when a new model does not match either.
    """
    is_wan = model.startswith("wan")
    route = path or (
        "/services/aigc/text2image/image-synthesis" if is_wan
        else "/services/aigc/multimodal-generation/generation"
    )

    if is_wan:
        body = {"model": model, "input": {"prompt": prompt},
                "parameters": {"n": n, **({"size": size} if size else {})}}
        if negative:
            body["input"]["negative_prompt"] = negative
        if images:
            body["input"]["ref_img"] = images[0]
    else:
        content: list[dict] = [{"image": i} for i in (images or [])]
        content.append({"text": prompt})
        body = {"model": model, "input": {"messages": [{"role": "user", "content": content}]},
                "parameters": {**({"size": size} if size else {}), **({"n": n} if n > 1 else {})}}
        if negative:
            body["parameters"]["negative_prompt"] = negative
    if extra:
        body.setdefault("parameters", {}).update(extra)

    if is_wan:
        task_id = c.submit_task(route, body)
        out = c.wait_task(task_id, on_poll=on_poll)
        urls = [r.get("url") for r in (out.get("results") or []) if r.get("url")]
    else:
        payload = c.post(f"{c.cfg.dashscope_base}{route}", body)
        out = payload.get("output") or {}
        urls = _dig_urls(out)
        c.record("image", model, payload.get("usage"), {"n": len(urls)})
    if is_wan:
        c.record("image", model, None, {"n": len(urls), "task_id": task_id})
    return {"urls": urls, "raw": out}


def _dig_urls(out: dict) -> list[str]:
    """Pull image/audio urls out of a multimodal-generation response."""
    urls = []
    for ch in out.get("choices") or []:
        for part in (ch.get("message") or {}).get("content") or []:
            for k in ("image", "url", "audio"):
                v = part.get(k) if isinstance(part, dict) else None
                if isinstance(v, str) and v.startswith("http"):
                    urls.append(v)
                elif isinstance(v, dict) and v.get("url"):
                    urls.append(v["url"])
    for r in out.get("results") or []:
        if r.get("url"):
            urls.append(r["url"])
    if isinstance(out.get("audio"), dict) and out["audio"].get("url"):
        urls.append(out["audio"]["url"])
    return urls


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------

def video_generate(c: Client, prompt: str, *, model: str = "wan2.6-i2v-flash",
                   image: str | None = None, audio: str | None = None,
                   resolution: str = "720P", duration: int | None = None,
                   prompt_extend: bool = True, watermark: bool = False,
                   path: str | None = None, extra: dict | None = None, on_poll=None,
                   wait: bool = True) -> dict:
    """Text-to-video, or image-to-video when `image` is a public URL.

    Always async. Typical 1-5 min. The task id stays valid for 24h, so a
    timeout here loses nothing — re-poll with `ms task <id>`.
    """
    inp: dict = {"prompt": prompt}
    if image:
        inp["img_url"] = image
    if audio:
        inp["audio_url"] = audio
    params = {"resolution": resolution, "prompt_extend": prompt_extend, "watermark": watermark}
    if duration:
        params["duration"] = duration
    if extra:
        params.update(extra)

    route = path or "/services/aigc/video-generation/video-synthesis"
    task_id = c.submit_task(route, {"model": model, "input": inp, "parameters": params})
    c.record("video", model, None, {"task_id": task_id})
    if not wait:
        return {"url": None, "task_id": task_id, "status": "SUBMITTED", "raw": None}
    out = c.wait_task(task_id, on_poll=on_poll)
    return {"url": out.get("video_url"), "task_id": task_id, "raw": out}


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def transcribe(c: Client, urls: list[str], *, model: str = "qwen3-asr-flash-filetrans",
               language: str | None = None, path: str | None = None,
               extra: dict | None = None, on_poll=None, wait: bool = True) -> dict:
    """Async file transcription. Takes PUBLIC URLs only — no local upload path."""
    for u in urls:
        if not u.startswith(("http://", "https://")):
            raise ValueError(
                f"{u!r} is not a URL. File transcription accepts publicly reachable "
                "URLs only; upload to OSS (or any public host) first."
            )
    params = dict(extra or {})
    if language:
        params["language_hints"] = [language]
    route = path or "/services/audio/asr/transcription"
    task_id = c.submit_task(route, {"model": model, "input": {"file_urls": urls}, "parameters": params})
    c.record("asr", model, None, {"files": len(urls), "task_id": task_id})
    if not wait:
        return {"results": [], "task_id": task_id, "status": "SUBMITTED", "raw": None}
    out = c.wait_task(task_id, on_poll=on_poll)
    return {"results": out.get("results") or [], "task_id": task_id, "raw": out}


def speak(c: Client, text: str, *, model: str = "qwen3-tts-flash", voice: str = "Cherry",
          language: str | None = None, path: str | None = None, extra: dict | None = None) -> dict:
    """Non-streaming TTS. Returns a URL to the rendered audio.

    Verified live: qwen3-tts-flash on the multimodal-generation route answers
    synchronously with output.audio.url. The tts/text-to-speech route rejects
    this shape, and qwen-audio-3.0-tts-plus is not enabled on this workspace.
    """
    params = {"voice": voice, **(extra or {})}
    if language:
        params["language_type"] = language
    route = path or "/services/aigc/multimodal-generation/generation"
    payload = c.post(
        f"{c.cfg.dashscope_base}{route}",
        {"model": model, "input": {"text": text}, "parameters": params},
    )
    out = payload.get("output") or {}
    urls = _dig_urls(out)
    c.record("tts", model, payload.get("usage"), {"chars": len(text)})
    return {"url": urls[0] if urls else None, "raw": out}


# --------------------------------------------------------------------------
# files + batch
# --------------------------------------------------------------------------

def file_upload(c: Client, path: Path, purpose: str = "fine-tune") -> dict:
    return c.upload(f"{c.cfg.openai_base}/files", path, {"purpose": purpose})


def file_list(c: Client) -> list[dict]:
    return (c.get(f"{c.cfg.openai_base}/files", timeout=60)).get("data", [])


def file_delete(c: Client, file_id: str) -> dict:
    return c.delete(f"{c.cfg.openai_base}/files/{file_id}", timeout=60)


def batch_create(c: Client, file_id: str, *, endpoint: str = "/v1/chat/completions",
                 window: str = "24h") -> dict:
    """Batch inference bills at 50% of real-time for the same model."""
    return c.post(f"{c.cfg.openai_base}/batches",
                  {"input_file_id": file_id, "endpoint": endpoint, "completion_window": window})


def batch_get(c: Client, batch_id: str) -> dict:
    return c.get(f"{c.cfg.openai_base}/batches/{batch_id}", timeout=60)


def batch_list(c: Client, limit: int = 20) -> list[dict]:
    return (c.get(f"{c.cfg.openai_base}/batches?limit={limit}", timeout=60)).get("data", [])


def batch_cancel(c: Client, batch_id: str) -> dict:
    return c.post(f"{c.cfg.openai_base}/batches/{batch_id}/cancel")


def file_content(c: Client, file_id: str) -> str:
    resp = c.request("GET", f"{c.cfg.openai_base}/files/{file_id}/content", raw=True, timeout=600)
    with resp:
        return resp.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# fine-tuning
# --------------------------------------------------------------------------

def tune_create(c: Client, model: str, training_file_ids: list[str], *,
                validation_file_ids: list[str] | None = None,
                training_type: str = "efficient_sft",
                hyper: dict | None = None) -> dict:
    """Create a tuning job.

    training_type: sft | efficient_sft (LoRA) | cpt | dpo_full | dpo_lora
    One job runs at a time per account; the rest sit in QUEUING.
    """
    body = {
        "model": model,
        "training_file_ids": training_file_ids,
        "training_type": training_type,
        "hyper_parameters": hyper or {"n_epochs": 3, "batch_size": 16, "max_length": 8192},
    }
    if validation_file_ids:
        body["validation_file_ids"] = validation_file_ids
    return c.post(f"{c.cfg.dashscope_base}/fine-tunes", body)


def tune_list(c: Client) -> dict:
    return c.get(f"{c.cfg.dashscope_base}/fine-tunes", timeout=60)


def tune_get(c: Client, job_id: str) -> dict:
    return c.get(f"{c.cfg.dashscope_base}/fine-tunes/{job_id}", timeout=60)


def tune_logs(c: Client, job_id: str, offset: int = 0, line: int = 1000) -> dict:
    return c.get(f"{c.cfg.dashscope_base}/fine-tunes/{job_id}/logs?offset={offset}&line={line}", timeout=120)


def tune_checkpoints(c: Client, job_id: str) -> dict:
    return c.get(f"{c.cfg.dashscope_base}/fine-tunes/{job_id}/checkpoints", timeout=60)


def tune_deploy(c: Client, job_id: str, checkpoint_id: str, model_name: str) -> dict:
    """Publish a checkpoint as a callable model instance id."""
    return c.get(
        f"{c.cfg.dashscope_base}/fine-tunes/{job_id}/export/{checkpoint_id}?model_name={model_name}",
        timeout=300,
    )


def tune_cancel(c: Client, job_id: str) -> dict:
    return c.post(f"{c.cfg.dashscope_base}/fine-tunes/{job_id}/cancel")


def tune_delete(c: Client, job_id: str) -> dict:
    return c.delete(f"{c.cfg.dashscope_base}/fine-tunes/{job_id}", timeout=60)


def tuned_generate(c: Client, model_instance_id: str, prompt: str, **kw) -> dict:
    """Call a deployed fine-tuned model. It lives on the native text-generation route,
    not on /chat/completions."""
    payload = c.post(
        f"{c.cfg.dashscope_base}/services/aigc/text-generation/generation",
        {"model": model_instance_id,
         "input": {"messages": [{"role": "user", "content": prompt}]},
         "parameters": {"result_format": "message", **kw}},
    )
    c.record("tuned", model_instance_id, payload.get("usage"))
    return payload
