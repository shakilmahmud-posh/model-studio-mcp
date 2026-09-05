"""Credential and endpoint resolution for Alibaba Cloud Model Studio.

Three API surfaces share one host and one key:

    /compatible-mode/v1   OpenAI-compatible  (chat, embeddings, files, batches, models)
    /api/v1               DashScope native   (image, video, ASR, TTS, rerank, fine-tunes)
    /apps/anthropic       Anthropic-compatible (clients append /v1/messages)

The API key is read from the environment or from a credentials file and is never
printed, logged, or written to a trace. Only its last 4 characters are ever shown.
"""
from __future__ import annotations

import os
from pathlib import Path

def _default_cred_file() -> Path:
    """Where the credentials file lives, in order of preference.

    MODEL_STUDIO_ENV wins outright. Otherwise ~/.model-studio.env, which is
    where it belongs on any machine. ~/dev/.model-studio.env is checked last
    and only if it exists, so an installation that predates this ordering keeps
    working without being told to move anything.
    """
    override = os.environ.get("MODEL_STUDIO_ENV")
    if override:
        return Path(override)
    home = Path.home() / ".model-studio.env"
    if home.exists():
        return home
    legacy = Path.home() / "dev" / ".model-studio.env"
    if legacy.exists():
        return legacy
    return home                      # the path we tell people to create


CRED_FILE = _default_cred_file()

# Fallback host when no workspace id is configured. Alibaba is migrating callers
# off this to workspace-scoped hosts, so it is a fallback and not the default.
GLOBAL_INTL_HOST = "https://dashscope-intl.aliyuncs.com"

DEFAULT_REGION = "ap-southeast-1"


class ConfigError(RuntimeError):
    """Raised when credentials or endpoints cannot be resolved."""


def _read_cred_file() -> dict:
    if not CRED_FILE.exists():
        return {}
    out = {}
    for line in CRED_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Config:
    __slots__ = ("api_key", "workspace", "region", "host", "default_model", "ledger_path")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    # -- surfaces ---------------------------------------------------------
    @property
    def openai_base(self) -> str:
        """OpenAI-compatible surface."""
        return f"{self.host}/compatible-mode/v1"

    @property
    def dashscope_base(self) -> str:
        """DashScope native surface."""
        return f"{self.host}/api/v1"

    @property
    def anthropic_base(self) -> str:
        """Anthropic-compatible surface. Clients append /v1/messages themselves."""
        return f"{self.host}/apps/anthropic"

    @property
    def key_tail(self) -> str:
        return f"...{self.api_key[-4:]}" if self.api_key else "(none)"

    def describe(self) -> dict:
        """Safe-to-print summary. Never includes the key itself."""
        return {
            "host": self.host,
            "region": self.region,
            "workspace": self.workspace or "(none — global intl fallback)",
            "key": self.key_tail,
            "openai_base": self.openai_base,
            "dashscope_base": self.dashscope_base,
            "anthropic_base": self.anthropic_base,
            "default_model": self.default_model,
            "ledger": str(self.ledger_path),
        }


def load(require_key: bool = True) -> Config:
    """Resolve config. Environment wins over the credentials file."""
    f = _read_cred_file()

    def pick(name: str, default: str = "") -> str:
        return os.environ.get(name) or f.get(name) or default

    api_key = pick("DASHSCOPE_API_KEY")
    if require_key and not api_key:
        raise ConfigError(
            f"no DASHSCOPE_API_KEY.\n"
            f"  Create the key at https://modelstudio.console.alibabacloud.com (Singapore region),\n"
            f"  then put it in {CRED_FILE} as DASHSCOPE_API_KEY=sk-...  and chmod 600 that file.\n"
            f"  Template: {CRED_FILE}.example"
        )

    workspace = pick("DASHSCOPE_WORKSPACE_ID")
    region = pick("DASHSCOPE_REGION", DEFAULT_REGION)

    host = pick("DASHSCOPE_HOST")
    if not host:
        host = f"https://{workspace}.{region}.maas.aliyuncs.com" if workspace else GLOBAL_INTL_HOST
    host = host.rstrip("/")

    ledger = Path(pick("MODEL_STUDIO_LEDGER", str(Path.home() / "dev" / "model-studio" / "state" / "usage.jsonl")))

    return Config(
        api_key=api_key,
        workspace=workspace,
        region=region,
        host=host,
        default_model=pick("DASHSCOPE_DEFAULT_MODEL", "qwen-plus"),
        ledger_path=ledger,
    )
