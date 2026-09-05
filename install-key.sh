#!/usr/bin/env bash
# Move a Model Studio API key from a downloaded file into the credentials file.
#
#   bash install-key.sh ~/Downloads/<downloaded-file>
#   bash install-key.sh --inspect ~/Downloads/<downloaded-file>
#
# Console exports are frequently UTF-16 or GB18030 rather than UTF-8, so the key
# is looked for through several decodings before giving up.
#
# The key never reaches stdout, a log, or a Claude session. Success prints only
# the key's length and last 4 characters. --inspect prints only structure:
# encoding, sizes, and which decoding contains a key. Never file content.
set -euo pipefail

INSPECT=0
if [[ "${1:-}" == "--inspect" ]]; then INSPECT=1; shift; fi

SRC="${1:-}"
# MODEL_STUDIO_ENV wins; then an existing ~/dev/.model-studio.env (older
# installs); otherwise ~/.model-studio.env, which is where it belongs.
if [[ -n "${MODEL_STUDIO_ENV:-}" ]]; then
  ENV_FILE="$MODEL_STUDIO_ENV"
elif [[ -f "$HOME/dev/.model-studio.env" ]]; then
  ENV_FILE="$HOME/dev/.model-studio.env"
else
  ENV_FILE="$HOME/.model-studio.env"
fi
TEMPLATE="${ENV_FILE}.example"
KEY_RE='sk-[A-Za-z0-9_-]{16,}'

if [[ -z "$SRC" ]]; then
  echo "usage: bash install-key.sh [--inspect] <path-to-downloaded-key-file>" >&2
  exit 2
fi
[[ -f "$SRC" ]] || { echo "error: no such file: $SRC" >&2; exit 2; }

# Emit the file's text through one decoding. Never printed — only piped to grep.
decode() {
  case "$1" in
    raw)      cat "$SRC" ;;
    nulls)    tr -d '\000' < "$SRC" ;;                       # crude UTF-16 -> ASCII
    utf16)    iconv -f UTF-16   -t UTF-8 "$SRC" 2>/dev/null ;;
    utf16le)  iconv -f UTF-16LE -t UTF-8 "$SRC" 2>/dev/null ;;
    utf16be)  iconv -f UTF-16BE -t UTF-8 "$SRC" 2>/dev/null ;;
    gb18030)  iconv -f GB18030  -t UTF-8 "$SRC" 2>/dev/null ;;
    latin1)   iconv -f ISO-8859-1 -t UTF-8 "$SRC" 2>/dev/null ;;
  esac
}
DECODINGS=(raw nulls utf16 utf16le utf16be gb18030 latin1)

if (( INSPECT )); then
  echo "file     : $(basename "$SRC")"
  echo "type     : $(file -b "$SRC")"
  echo "bytes    : $(wc -c < "$SRC" | tr -d ' ')"
  echo "lines    : $(wc -l < "$SRC" | tr -d ' ')"
  echo
  echo "decoding      sk-matches   'sk' substrings   masked(*)"
  for d in "${DECODINGS[@]}"; do
    txt="$(decode "$d" || true)"
    full=$(printf '%s' "$txt" | grep -oE "$KEY_RE" | wc -l | tr -d ' ')
    loose=$(printf '%s' "$txt" | grep -o 'sk' | wc -l | tr -d ' ')
    stars=$(printf '%s' "$txt" | grep -o '\*' | wc -l | tr -d ' ')
    printf "  %-12s %-12s %-17s %s\n" "$d" "$full" "$loose" "$stars"
  done
  echo
  echo "A 'sk' substring with 0 full matches means the key is there but shorter"
  echo "or masked. Many '*' means the export masked it and only the creation"
  echo "dialog ever had the plaintext."
  exit 0
fi

KEY=""
USED=""
for d in "${DECODINGS[@]}"; do
  KEY="$(decode "$d" | grep -oE "$KEY_RE" | head -1 || true)"
  if [[ -n "$KEY" ]]; then USED="$d"; break; fi
done

if [[ -z "$KEY" ]]; then
  echo "error: no key matching sk-... found in $SRC" >&2
  echo "       Tried: ${DECODINGS[*]}" >&2
  echo "       Run for a safe structural report (prints no file content):" >&2
  echo "         bash install-key.sh --inspect \"$SRC\"" >&2
  exit 1
fi

[[ -f "$ENV_FILE" ]] || cp "$TEMPLATE" "$ENV_FILE"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
awk -v key="$KEY" '
  /^DASHSCOPE_API_KEY=/ { print "DASHSCOPE_API_KEY=" key; found=1; next }
  { print }
  END { if (!found) print "DASHSCOPE_API_KEY=" key }
' "$ENV_FILE" > "$TMP"
cat "$TMP" > "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "wrote key to $ENV_FILE"
echo "  decoding : $USED"
echo "  length   : ${#KEY}"
echo "  tail     : ...${KEY: -4}"
echo "  perms    : $(stat -f '%Sp' "$ENV_FILE")"
echo
echo "The downloaded file is still a plaintext secret on disk. Remove it:"
echo "  rm -P \"$SRC\""
echo
echo "Then prove it works:"
echo "  cd ~/dev/model-studio && ./ms doctor"
