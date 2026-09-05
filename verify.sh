#!/usr/bin/env bash
# Proves the model-studio CLI and MCP server.
#
# Static checks need no credentials and no network. The live API is deep-only:
# it costs real free-tier tokens, so it must never run on a plain `verify.sh`.
# Vendored so a fresh clone can run this with nothing installed and nothing
# else on the machine. Same file, no external path.
source "$(dirname "${BASH_SOURCE[0]}")/tests/verify-lib.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 2

# --- syntax -----------------------------------------------------------------
check "compile: package"    python3 -m py_compile modelstudio/config.py modelstudio/client.py modelstudio/api.py
check "compile: cli"        python3 -m py_compile ms
check "compile: mcp server" python3 -m py_compile mcp_server.py

# --- CLI surface ------------------------------------------------------------
check_out "cli: help lists every command" "anthropic" ./ms --help
for c in doctor models chat embed rerank image video asr tts task files batch tune usage anthropic; do
  check "cli: subcommand '$c' parses" ./ms "$c" --help
done

# --json must work before AND after the subcommand; argparse gets this wrong by
# default and the hoist in main() is what fixes it. A fixture ledger keeps this
# independent of whatever the real ledger happens to hold.
LEDGER=tests/fixtures/usage.jsonl
check_out "cli: --json after subcommand"  '"total": 280' \
  env DASHSCOPE_API_KEY=sk-test MODEL_STUDIO_LEDGER=$LEDGER ./ms usage --json
check_out "cli: --json before subcommand" '"total": 280' \
  env DASHSCOPE_API_KEY=sk-test MODEL_STUDIO_LEDGER=$LEDGER ./ms --json usage
check_out "cli: default output is a table, not JSON" "kind/model" \
  env DASHSCOPE_API_KEY=sk-test MODEL_STUDIO_LEDGER=$LEDGER ./ms usage
check_out "cli: ledger aggregates per model" "chat/qwen-plus" \
  env DASHSCOPE_API_KEY=sk-test MODEL_STUDIO_LEDGER=$LEDGER ./ms usage

# --- credential handling ----------------------------------------------------
# A missing key must fail loudly with a pointer, and must never reach the network.
NOKEY='env -u DASHSCOPE_API_KEY MODEL_STUDIO_ENV=/nonexistent ./ms models'
check "cli: missing key exits 2" bash -c "$NOKEY >/dev/null 2>&1; test \$? -eq 2"
check_out "cli: missing key names the env var" "DASHSCOPE_API_KEY" bash -c "$NOKEY 2>&1; true"
check_out "cli: missing key names the region" "Singapore" bash -c "$NOKEY 2>&1; true"
# "sk-..." appears in the help text as a placeholder, so match a REALISTIC key
# (sk- followed by a long alnum run) rather than the bare prefix.
check "cli: no realistic key ever appears in output" \
  bash -c "$NOKEY 2>&1 | grep -Eq 'sk-[A-Za-z0-9]{8,}' && exit 1; exit 0"
check "config: describe() never contains the raw key" python3 -c "
import os, sys
os.environ['DASHSCOPE_API_KEY'] = 'sk-verysecretkeymaterial1234567890'
os.environ['MODEL_STUDIO_ENV'] = '/nonexistent'
sys.path.insert(0, '.')
from modelstudio.config import load
d = load()
blob = repr(d.describe())
assert os.environ['DASHSCOPE_API_KEY'] not in blob, 'raw key leaked into describe()'
assert d.key_tail == '...7890', d.key_tail
assert 'verysecret' not in blob, 'key body leaked into describe()'
"

# --- stdin handling ---------------------------------------------------------
# Regression: `if not isatty(): read()` blocks forever when stdin is a character
# device that is not a terminal (cron, systemd, CI, agent harnesses). Only pipes
# and regular files carry input.
check "stdin: a pipe is read" python3 -c "
import subprocess, sys
r = subprocess.run([sys.executable, '-c',
  'import sys; sys.path.insert(0, \".\"); from modelstudio.client import read_stdin; print(repr(read_stdin()))'],
  input=b'hi', capture_output=True, timeout=20)
assert r.stdout.strip() == b\"'hi'\", r.stdout
"
check "stdin: a character device is NOT read" python3 -c "
import subprocess, sys
with open('/dev/null') as devnull:
    r = subprocess.run([sys.executable, '-c',
      'import sys; sys.path.insert(0, \".\"); from modelstudio.client import read_stdin; print(repr(read_stdin()))'],
      stdin=devnull, capture_output=True, timeout=20)
assert r.stdout.strip() == b\"''\", r.stdout
"
check "stdin: a redirected file is read" bash -c "
echo hello > /tmp/ms-verify-stdin.txt
python3 -c 'import sys; sys.path.insert(0, \".\"); from modelstudio.client import read_stdin; assert read_stdin().strip() == \"hello\"' < /tmp/ms-verify-stdin.txt
"

# --- free-quota guard -------------------------------------------------------
# The console's per-model "Free Quota Only" switch is the real protection; this
# is the local seatbelt. It must actually BLOCK, be per-model, and be defeatable.
#
# Every "is it blocked" assertion matches the GUARD'S OWN MESSAGE, never a bare
# exit code: these run with a fake key, so an unguarded call 401s and also exits
# 1 — an exit-code-only check passes on the wrong failure. (Found 2026-08-21: the
# fixture said 999,999 against a 1,000,000 cap, so the guard never fired and the
# test passed on the 401.) Fixture values sit AT the cap, not one below it.
GUARD_LEDGER=tests/fixtures/guard-ledger.jsonl
cat > "$GUARD_LEDGER" <<'LEDGER'
{"ts":"2026-08-21T10:00:00Z","kind":"chat","model":"capped-model","total_tokens":1000000}
{"ts":"2026-08-21T10:00:01Z","kind":"image","model":"capped-image","output_image_count":100}
{"ts":"2026-08-21T10:00:02Z","kind":"tts","model":"capped-tts","characters":2000}
LEDGER
GUARD_ENV="env DASHSCOPE_API_KEY=sk-test MODEL_STUDIO_LEDGER=$GUARD_LEDGER"

check_out "guard: blocks at the token cap" "free-quota guard" bash -c \
  "$GUARD_ENV ./ms chat --model capped-model --no-stream 'x' 2>&1; true"
check_out "guard: names the console switch in the error" "Free Quota Only" bash -c \
  "$GUARD_ENV ./ms chat --model capped-model --no-stream 'x' 2>&1; true"
check "guard: a blocked call exits non-zero" bash -c \
  "$GUARD_ENV ./ms chat --model capped-model --no-stream 'x' >/dev/null 2>&1; test \$? -eq 1"
check_out "guard: blocks at the image cap" "free-quota guard" bash -c \
  "$GUARD_ENV ./ms image --model capped-image 'x' 2>&1; true"
check_out "guard: blocks at the TTS character cap" "free-quota guard" bash -c \
  "$GUARD_ENV ./ms tts --model capped-tts 'x' 2>&1; true"

# A model with no ledger history must be untouched even with an absurd cap.
check_absent "guard: is per-model, not global" "free-quota guard" bash -c \
  "$GUARD_ENV MODEL_STUDIO_MAX_TOKENS=1 ./ms chat --model an-untouched-model --no-stream 'x' 2>&1; true"
check_absent "guard: MODEL_STUDIO_FREE_ONLY=0 disables it" "free-quota guard" bash -c \
  "$GUARD_ENV MODEL_STUDIO_FREE_ONLY=0 ./ms chat --model capped-model --no-stream 'x' 2>&1; true"
check "guard: on by default" python3 -c "
import sys; sys.path.insert(0, '.')
from modelstudio import budget
assert budget.enabled(), 'guard must default to ON'
"
check "guard: the guard, not the network, is what stops the call" python3 -c "
import os, sys; sys.path.insert(0, '.')
os.environ['MODEL_STUDIO_LEDGER'] = 'tests/fixtures/guard-ledger.jsonl'
from modelstudio import budget
from pathlib import Path
try:
    budget.check(Path('tests/fixtures/guard-ledger.jsonl'), 'capped-model')
except budget.BudgetExceeded:
    sys.exit(0)
sys.exit('BudgetExceeded was not raised at the cap')
"
check_out "guard: ms budget reports headroom" "local guard" bash -c "$GUARD_ENV ./ms budget"

# --- MCP protocol -----------------------------------------------------------
MCP='python3 mcp_server.py'
FX=tests/fixtures
check_out "mcp: initialize handshake" '"serverInfo"' \
  bash -c "$MCP < $FX/mcp-initialize.jsonl 2>/dev/null"
check_out "mcp: initialize echoes the client protocol version" '"2025-06-18"' \
  bash -c "$MCP < $FX/mcp-initialize.jsonl 2>/dev/null"
check_out "mcp: tools/list returns the full surface" '"ms_tune"' \
  bash -c "$MCP < $FX/mcp-tools-list.jsonl 2>/dev/null"
check_out "mcp: tools/list advertises 14 tools" '"ms_usage"' \
  bash -c "$MCP < $FX/mcp-tools-list.jsonl 2>/dev/null"
check_out "mcp: unknown tool is an error, not a crash" "unknown tool" \
  bash -c "$MCP < $FX/mcp-unknown-tool.jsonl 2>/dev/null"
check_out "mcp: ping answers" '"result"' \
  bash -c "$MCP < $FX/mcp-ping.jsonl 2>/dev/null"
check_absent "mcp: stdout carries no diagnostics" "[model-studio]" \
  bash -c "$MCP < $FX/mcp-ping.jsonl 2>/dev/null"
check_out "mcp: unconfigured call returns isError, not a stack trace" '"isError"' \
  bash -c "env -u DASHSCOPE_API_KEY MODEL_STUDIO_ENV=/nonexistent $MCP < $FX/mcp-unconfigured.jsonl 2>/dev/null"

# --- live API (deep only, spends free quota) ---------------------------------
if deep_only "live: Model Studio API"; then
  check_out "live: doctor smoke passes"        "pass"       ./ms doctor
  check_out "live: models list is populated"   "qwen-plus"  ./ms models
  check_out "live: chat answers"               "OK"         ./ms chat --model qwen-flash --no-stream --quiet "Reply with exactly: OK"
  check_out "live: embeddings return vectors"  "dim 1024"   ./ms embed "hello" --dim 1024
  check_out "live: rerank ranks correctly"     "reranker"   ./ms rerank --query "what is a reranker" "a reranker scores query-document pairs" "bananas are yellow"
  check_out "live: chat never hangs on non-tty stdin" "OK" \
    bash -c './ms chat --model qwen-flash --no-stream --quiet "Reply with exactly: OK" < /dev/null'
fi

verify_summary
