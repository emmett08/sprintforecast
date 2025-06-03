#!/usr/bin/env bash
set -euo pipefail

[[ $# -lt 1 || $# -gt 2 ]] && { printf 'Usage: %s <tag> [asset]\n' "$0" >&2; exit 1; }

TAG="$1"
ASSET="${2:-}"

git add -A
git diff --cached --quiet || git commit -m "chore(release): ${TAG#v}"

git tag -a "$TAG" -m "$TAG"
git push origin HEAD --tags

if [[ -n "$ASSET" ]]; then
  gh release create "$TAG" -t "$TAG" -n "Release ${TAG#v}" "$ASSET"
else
  gh release create "$TAG" -t "$TAG" -n "Release ${TAG#v}"
fi
printf 'Release %s created successfully.\n' "$TAG"
