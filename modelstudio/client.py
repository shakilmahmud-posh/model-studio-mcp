"""HTTP core for Model Studio. Stdlib only.

Handles the three things every call needs and none of the callers should repeat:
auth headers, the DashScope async-task dance, and error messages that name the
actual cause instead of echoing an HTTP code.
"""
from __future__ import annotations

import json
import mimetypes
import os
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import budget
from .config import Config

# The query API is capped at 20 QPS and the docs ask for >=15s between polls.
POLL_INTERVAL = 15
POLL_TIMEOUT = 1800  # 30 min — video jobs are the long pole
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        self.status, self.body, self.url = status, body, url
        super().__init__(self._explain())

    def _explain(self) -> str:
        host = urllib.parse.urlparse(self.url).netloc
        if "FreeTierOnly" in self.body:
            return ("free quota exhausted for this model, and 'Free Quota Only' is ON in the "
                    "console, so the call was BLOCKED rather than billed. That is the guard "
                    "working. Use a different model (quota is per-model) or turn the switch "
                    "off in the console to move to paid.")
        if "Arrearage" in self.body or "Overdue" in self.body:
            return f"HTTP {self.status}: account is in arrears — calls are blocked until settled.\n{self.body[:400]}"
        hint = {
            401: "key rejected. The API key and the endpoint must be in the SAME region — "
                 "a Beijing-region key always fails against ap-southeast-1.",
            403: "forbidden. Either the key lacks access to this model, or the model is not "
                 "enabled for this workspace.",
            404: "no such model or route at this host. Model ids differ per region — "
                 "run `ms models` for what this account can actually call.",
            429: "rate limited, or the free quota for this model is exhausted. "
                 "Free quota is per-model, so try another model to tell the two apart.",
            400: "malformed request. Check the model id supports this endpoint — "
                 "e.g. an image model cannot be called on /chat/completions.",
        }.get(self.status, "")
        return f"HTTP {self.status} from {host}{': ' + hint if hint else ''}\n{self.body[:900]}"


class Client:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # -- low level ---------------------------------------------------------
    def _headers(self, extra: dict | None = None, content_type: str | None = "application/json") -> dict:
        h = {"Authorization": f"Bearer {self.cfg.api_key}"}
        if content_type:
            h["Content-Type"] = content_type
        if extra:
            h.update(extra)
        return h

    def request(
        self,
        method: str,
        url: str,
        *,
        body: dict | None = None,
        headers: dict | None = None,
        timeout: int = 600,
        raw: bool = False,
    ):
        # Single choke point for the local free-quota guard: every Model Studio
        # request body names its model, so nothing can route around this.
        if body and body.get("model"):
            budget.check(self.cfg.ledger_path, body["model"])
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(headers), method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode("utf-8", errors="replace"), url) from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url}: {e.reason}") from None
        if raw:
            return resp
        with resp:
            payload = resp.read().decode("utf-8", errors="replace")
        return json.loads(payload) if payload else {}

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, body=None, **kw):
        return self.request("POST", url, body=body if body is not None else {}, **kw)

    def delete(self, url, **kw):
        return self.request("DELETE", url, **kw)

    # -- multipart upload --------------------------------------------------
    def upload(self, url: str, path: Path, fields: dict) -> dict:
        """multipart/form-data upload, built by hand to stay stdlib-only."""
        boundary = f"----ms{uuid.uuid4().hex}"
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts: list[bytes] = []
        for k, v in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        payload = b"".join(parts)

        req = urllib.request.Request(
            url,
            data=payload,
            headers=self._headers(content_type=f"multipart/form-data; boundary={boundary}"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode("utf-8", errors="replace"), url) from None

    # -- async task pattern -------------------------------------------------
    def submit_task(self, path: str, body: dict) -> str:
        """POST to a DashScope async endpoint, return the task_id."""
        url = f"{self.cfg.dashscope_base}{path}"
        out = self.post(url, body, headers={"X-DashScope-Async": "enable"})
        task_id = (out.get("output") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"no task_id in async response: {json.dumps(out)[:500]}")
        return task_id

    def get_task(self, task_id: str) -> dict:
        return self.get(f"{self.cfg.dashscope_base}/tasks/{task_id}", timeout=60)

    def wait_task(self, task_id: str, *, on_poll=None, interval: int = POLL_INTERVAL,
                  timeout: int = POLL_TIMEOUT) -> dict:
        """Block until the task reaches a terminal state. Returns the output block."""
        waited = 0
        while True:
            out = (self.get_task(task_id).get("output") or {})
            status = out.get("task_status", "UNKNOWN")
            if on_poll:
                on_poll(status, waited)
            if status in TERMINAL_STATES:
                if status != "SUCCEEDED":
                    msg = out.get("message") or out.get("code") or "no reason given"
                    if status == "UNKNOWN":
                        msg += " (task ids expire 24h after creation)"
                    raise RuntimeError(f"task {task_id} ended {status}: {msg}")
                return out
            if waited >= timeout:
                raise TimeoutError(
                    f"task {task_id} still {status} after {waited}s. "
                    f"It is not lost — poll it later with: ms task {task_id}"
                )
            time.sleep(interval)
            waited += interval

    # -- usage ledger -------------------------------------------------------
    def record(self, kind: str, model: str, usage: dict | None, extra: dict | None = None) -> None:
        """Append one line to the local usage ledger.

        Free quota is per-model and expires 90 days after activation, and the
        console gives no terminal-side view of the burn. This is that view.
        """
        try:
            p = Path(self.cfg.ledger_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            u = usage or {}
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kind": kind,
                "model": model,
                "input_tokens": u.get("prompt_tokens") or u.get("input_tokens"),
                "output_tokens": u.get("completion_tokens") or u.get("output_tokens"),
                "total_tokens": u.get("total_tokens"),
            }
            # Media services bill in their own units, not tokens: TTS reports
            # `characters`, image reports `output_image_count`. Recording only
            # tokens would make those calls look free in the ledger.
            for unit in ("characters", "output_image_count", "duration", "video_duration"):
                if u.get(unit) is not None:
                    row[unit] = u[unit]
            if extra:
                row.update(extra)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass  # never let bookkeeping break a working call


def download(url: str, dest: Path) -> Path:
    """Fetch a generated asset (image/video/audio) to disk. Result URLs are pre-signed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=900) as resp, dest.open("wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    return dest


def read_stdin() -> str:
    """Read piped input, or "" when nothing is piped.

    `isatty()` alone is NOT enough. Under cron, systemd, CI and agent harnesses
    stdin is often a character device that is not a terminal: isatty() returns
    False, so a naive `if not isatty(): read()` blocks forever waiting for an
    EOF that never comes. Only a pipe or a redirected file actually carries
    input, so gate on the file type instead.
    """
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError):
        return ""
    if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
        try:
            return sys.stdin.read()
        except (OSError, UnicodeDecodeError):
            return ""
    return ""


def data_uri(path: Path) -> str:
    """Inline a local image as a data: URI for vision calls."""
    import base64
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
