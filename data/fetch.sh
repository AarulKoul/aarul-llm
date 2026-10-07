#!/usr/bin/env bash
# Fetch TinyStories from Hugging Face. huggingface.co itself resets connections
# intermittently on this network, but its CDN is fine, so: retry until we get
# the signed CDN redirect, then download (resumably) from the CDN.
set -u
cd "$(dirname "$0")"
for f in TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt; do
  for attempt in $(seq 1 40); do
    loc=$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 15 \
      "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$f")
    [ -n "$loc" ] && curl -sS --retry 5 -C - -o "$f" "$loc" && break
    sleep 2
  done
  ls -la "$f"
done
