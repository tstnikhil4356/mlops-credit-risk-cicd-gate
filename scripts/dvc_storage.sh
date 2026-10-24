#!/usr/bin/env bash
# Manage the DVC remote, which lives on the orphan branch `dvc-storage` and is
# checked out as a worktree at .dvc-storage/ (git-ignored on main).
#   scripts/dvc_storage.sh pull   # fresh clone: fetch storage branch + dvc pull
#   scripts/dvc_storage.sh push   # after `dvc add`: dvc push + commit/push storage branch
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DVC="${DVC:-dvc}"; command -v "$DVC" >/dev/null || DVC=.venv/bin/dvc

ensure_worktree() {
  if [ ! -e .dvc-storage/.git ]; then
    git fetch origin dvc-storage:dvc-storage 2>/dev/null || git fetch origin dvc-storage 2>/dev/null || true
    git show-ref --verify --quiet refs/heads/dvc-storage || git branch dvc-storage origin/dvc-storage
    git worktree add .dvc-storage dvc-storage
  fi
}

case "${1:-pull}" in
  pull)
    ensure_worktree
    (cd .dvc-storage && git pull --ff-only origin dvc-storage 2>/dev/null || true)
    "$DVC" pull
    ;;
  push)
    ensure_worktree
    "$DVC" push
    (cd .dvc-storage && git add -A && { git diff --cached --quiet || git commit -m "storage: dvc push $(date -u +%Y-%m-%dT%H:%MZ)"; })
    git push origin dvc-storage 2>/dev/null || echo "(no 'origin' remote yet: storage committed locally)"
    ;;
  *) echo "usage: $0 pull|push" >&2; exit 2 ;;
esac
