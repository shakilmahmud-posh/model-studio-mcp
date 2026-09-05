#!/usr/bin/env bash
# verify-lib.sh — the check DSL sourced by every <project>/.claude/verify.sh
#
# Contract for a project verify script:
#   source "$(dirname "${BASH_SOURCE[0]}")/tests/verify-lib.sh"
#   check "typecheck" npm run typecheck
#   check_out "build emits an app bundle" "index" ls dist/assets
#   verify_summary          # prints the table, sets the exit code
#
# WHY THIS EXISTS
# The factory can prove itself in one command (factory-test.sh, 487 assertions).
# No project could. Every project's verification was prose in its CLAUDE.md,
# re-derived by every session and only as good as whoever remembered to run it.
# Boris Cherny's framing (YC, 2026-08): the job is to make it possible for the
# model to verify its work along the way — you cannot safely hand over a harder
# task and step back unless the worker can prove it finished.
#
# THE ONE RULE: EXIT CODE IS TRUTH.
# `check` judges on exit status and nothing else. `check_out` requires exit 0
# AND the substring, in that order — a command that dies with a stack trace
# containing the word you grepped for is a FAILURE, not a pass. This is the
# factory's most-repeated own-goal (memory feedback_grep_on_failed_command_false_pass,
# reference_preview_deploy_green_check_lies): a green check that cannot go red
# is not a check, it is decoration.

set -uo pipefail   # deliberately NOT -e: a failing check must be recorded, not abort the run

VERIFY_PASS=0
VERIFY_FAIL=0
VERIFY_SKIP=0
VERIFY_ROWS=()
VERIFY_FAILED_NAMES=()

# Per-check timeout (seconds). A hung build must fail the run, never hang the session.
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-300}"

# Depth: "static" (default, fast + always safe) or "deep" (adds runtime/browser).
VERIFY_DEPTH="${VERIFY_DEPTH:-static}"

_c_green=$'\033[32m'; _c_red=$'\033[31m'; _c_dim=$'\033[2m'; _c_off=$'\033[0m'
[ -t 1 ] || { _c_green=""; _c_red=""; _c_dim=""; _c_off=""; }

# Resolve a timeout binary. macOS has neither `timeout` nor `gtimeout` by default;
# without one we still run the check, just unbounded — degraded, and we say so
# rather than silently pretending the timeout is being enforced.
_verify_timeout_bin() {
  if command -v timeout >/dev/null 2>&1; then echo "timeout"
  elif command -v gtimeout >/dev/null 2>&1; then echo "gtimeout"
  else echo ""; fi
}

_verify_run() {
  local tb; tb=$(_verify_timeout_bin)
  if [ -n "$tb" ]; then "$tb" "$VERIFY_TIMEOUT" "$@"; else "$@"; fi
}

_verify_record() {
  local status="$1" name="$2" detail="${3:-}"
  case "$status" in
    PASS) VERIFY_PASS=$((VERIFY_PASS+1)); VERIFY_ROWS+=("${_c_green}PASS${_c_off}  ${name}") ;;
    FAIL) VERIFY_FAIL=$((VERIFY_FAIL+1)); VERIFY_FAILED_NAMES+=("$name")
          VERIFY_ROWS+=("${_c_red}FAIL${_c_off}  ${name}${detail:+  ${_c_dim}(${detail})${_c_off}}") ;;
    SKIP) VERIFY_SKIP=$((VERIFY_SKIP+1)); VERIFY_ROWS+=("${_c_dim}SKIP  ${name}${detail:+  (${detail})}${_c_off}") ;;
  esac
}

# check "<name>" <command...>
# PASS iff the command exits 0. Output is captured and only shown on failure —
# a green run stays readable, a red run shows you the last 25 lines of why.
check() {
  local name="$1"; shift
  local out rc
  out=$(_verify_run "$@" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    _verify_record PASS "$name"
  else
    local why="exit ${rc}"
    [ $rc -eq 124 ] && why="TIMEOUT after ${VERIFY_TIMEOUT}s"
    _verify_record FAIL "$name" "$why"
    printf '%s\n' "--- ${name}: last 25 lines ---" >&2
    printf '%s\n' "$out" | tail -25 >&2
    printf '%s\n' "--- end ${name} ---" >&2
  fi
  return 0
}

# check_out "<name>" "<expected-substring>" <command...>
# PASS iff the command exits 0 AND its output contains the substring.
# The exit-code test comes FIRST and is non-negotiable: greping the output of a
# command that already failed is how a false PASS gets manufactured.
check_out() {
  local name="$1" expect="$2"; shift 2
  local out rc
  out=$(_verify_run "$@" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    local why="exit ${rc} (substring never evaluated)"
    [ $rc -eq 124 ] && why="TIMEOUT after ${VERIFY_TIMEOUT}s"
    _verify_record FAIL "$name" "$why"
    printf '%s\n' "--- ${name}: last 25 lines ---" >&2
    printf '%s\n' "$out" | tail -25 >&2
    return 0
  fi
  if printf '%s' "$out" | grep -qF -- "$expect"; then
    _verify_record PASS "$name"
  else
    _verify_record FAIL "$name" "missing: ${expect}"
  fi
  return 0
}

# check_absent "<name>" "<forbidden-substring>" <command...>
# PASS iff the command exits 0 AND its output does NOT contain the substring.
# For "no console errors", "no TODO markers shipped", "no demo picker in the bundle".
check_absent() {
  local name="$1" forbid="$2"; shift 2
  local out rc
  out=$(_verify_run "$@" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    _verify_record FAIL "$name" "exit ${rc} (absence never evaluated)"
    printf '%s\n' "$out" | tail -25 >&2
    return 0
  fi
  if printf '%s' "$out" | grep -qF -- "$forbid"; then
    _verify_record FAIL "$name" "found: ${forbid}"
  else
    _verify_record PASS "$name"
  fi
  return 0
}

# deep_only "<name>" — guard for checks that need a running app or a browser.
# Returns 0 when deep mode is on, else records a SKIP and returns 1:
#   deep_only "landing page renders" || return 0
deep_only() {
  local name="$1"
  if [ "$VERIFY_DEPTH" = "deep" ]; then return 0; fi
  _verify_record SKIP "$name" "static mode; run with --deep"
  return 1
}

# require_cmd "<binary>" — record a SKIP (not a FAIL) for a missing local tool.
# A machine without `psql` should not turn a code verification red.
require_cmd() {
  if command -v "$1" >/dev/null 2>&1; then return 0; fi
  _verify_record SKIP "$1 checks" "$1 not on PATH"
  return 1
}

verify_summary() {
  echo ""
  printf '%s\n' "${VERIFY_ROWS[@]}"
  echo ""
  local tb; tb=$(_verify_timeout_bin)
  [ -z "$tb" ] && echo "${_c_dim}note: no timeout binary found (brew install coreutils) — checks ran unbounded${_c_off}"
  if [ "$VERIFY_FAIL" -gt 0 ]; then
    echo "${_c_red}VERIFY FAILED${_c_off} — ${VERIFY_PASS} passed, ${VERIFY_FAIL} failed, ${VERIFY_SKIP} skipped"
    echo "failed: ${VERIFY_FAILED_NAMES[*]}"
    exit 1
  fi
  if [ "$VERIFY_PASS" -eq 0 ]; then
    # Zero checks ran. That is NOT a pass — it is an unprovable project, and
    # reporting it as green is exactly the lie this whole entrypoint exists to stop.
    echo "${_c_red}NOT PROVABLE${_c_off} — no checks executed (${VERIFY_SKIP} skipped)"
    exit 2
  fi
  echo "${_c_green}VERIFY GREEN${_c_off} — ${VERIFY_PASS} passed, ${VERIFY_SKIP} skipped (depth: ${VERIFY_DEPTH})"
  exit 0
}
