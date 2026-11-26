#!/usr/bin/env bash
# Download wetext text-normalization FSTs from modelscope.cn (repo pengzhendong/wetext)
# The modelscope SDK gets rate-limited/403'd by some proxies, so fetch over plain HTTPS.
set -euo pipefail
BASE="https://www.modelscope.cn/api/v1/models/pengzhendong/wetext/repo?Revision=master&FilePath="
DEST="pretrained_models/wetext"
for f in zh/tn/tagger.fst zh/tn/verbalizer.fst en/tn/tagger.fst en/tn/verbalizer.fst; do
  if [ -s "$DEST/$f" ]; then echo "  exists: $DEST/$f"; continue; fi
  mkdir -p "$DEST/$(dirname "$f")"
  echo "  downloading $f"
  for i in 1 2 3 4 5; do
    code=$(curl -s -o "$DEST/$f" -w "%{http_code}" "$BASE$f" || true)
    if [ "$code" = "200" ] && [ -s "$DEST/$f" ]; then break; fi
    echo "    retry $i (HTTP $code)"; sleep 10
  done
done
echo "wetext FSTs ready under $DEST"
