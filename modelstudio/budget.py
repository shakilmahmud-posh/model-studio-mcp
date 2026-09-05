"""Client-side spend guard.

The ONLY hard guarantee against charges is the per-model **Free Quota Only**
switch in the Model Studio console — that is enforced server-side and returns
`AllocationQuota.FreeTierOnly` instead of billing. This module is the second
layer: it refuses a call locally once the usage ledger says a model has burned
its free allowance.

Honest limit: the ledger only sees traffic that went through this tooling. Calls
made in the console playground, from another machine, or with any other client
are invisible here. Treat this as a seatbelt, never as the wall.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# Documented free allowances per model. Tokens are shared between input and
# output. TTS characters are documented as a range (2k-10k) so the default is
# the conservative end.
DEFAULTS = {
    "tokens": 1_000_000,
    "images": 100,
    "characters": 2_000,
    "seconds": 50,
}

FIELDS = (
    ("tokens", "total_tokens"),
    ("images", "output_image_count"),
    ("characters", "characters"),
    ("seconds", "duration"),
)


class BudgetExceeded(RuntimeError):
    pass


def enabled() -> bool:
    """On by default. Set MODEL_STUDIO_FREE_ONLY=0 to allow paid spend."""
    return os.environ.get("MODEL_STUDIO_FREE_ONLY", "1").lower() not in ("0", "false", "no")


def limits() -> dict:
    out = dict(DEFAULTS)
    for k in out:
        env = os.environ.get(f"MODEL_STUDIO_MAX_{k.upper()}")
        if env and env.isdigit():
            out[k] = int(env)
    return out


def spent(ledger: Path) -> dict:
    """Per-model totals from the local ledger."""
    agg: dict[str, dict] = defaultdict(lambda: {k: 0 for k, _ in FIELDS})
    if not Path(ledger).exists():
        return agg
    for line in Path(ledger).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = row.get("model")
        if not m:
            continue
        for key, field in FIELDS:
            agg[m][key] += row.get(field) or 0
    return agg


def check(ledger: Path, model: str) -> None:
    """Raise before a call that would push a model past its free allowance."""
    if not enabled() or not model:
        return
    used = spent(ledger).get(model)
    if not used:
        return
    lim = limits()
    for key, _ in FIELDS:
        if lim[key] and used[key] >= lim[key]:
            raise BudgetExceeded(
                f"local free-quota guard: {model} has used {used[key]:,} {key} "
                f"(cap {lim[key]:,}).\n"
                f"  This is a LOCAL stop, not the server's. It only counts calls made\n"
                f"  through this tooling — console and other clients are invisible to it.\n"
                f"  The real protection is the per-model 'Free Quota Only' switch in the\n"
                f"  console; check it is on for {model}.\n"
                f"  To proceed anyway (this may incur charges): MODEL_STUDIO_FREE_ONLY=0"
            )


def report(ledger: Path) -> list[dict]:
    """Per-model remaining allowance, for `ms budget`."""
    lim = limits()
    rows = []
    for model, used in sorted(spent(ledger).items()):
        row = {"model": model}
        for key, _ in FIELDS:
            if used[key]:
                row[key] = {"used": used[key], "cap": lim[key],
                            "left": max(0, lim[key] - used[key]),
                            "pct": round(100 * used[key] / lim[key], 2) if lim[key] else None}
        rows.append(row)
    return rows
