"""
Optional: automatically commit + push output files (predictions CSV +
metrics JSON) to GitHub after an evaluation run finishes.

Only activates if BOTH environment variables are set:
  GH_TOKEN  - a GitHub Personal Access Token (repo scope)
  GH_REPO   - "your-username/your-repo-name"

If either is missing, this silently no-ops — so the framework still runs
fine anywhere without git access (e.g. local debugging, Colab without a
token configured). Set these as environment variables in your Kaggle
notebook (from Kaggle Secrets) before running main.py.
"""

import os
import subprocess
from typing import List


def _run(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(f"$ {' '.join(c if c != os.environ.get('GH_TOKEN', object()) else '***' for c in cmd)}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result


def push_outputs_to_github(repo_root: str, file_paths: List[str], commit_message: str) -> None:
    gh_token = os.environ.get("GH_TOKEN")
    github_repo = os.environ.get("GH_REPO")

    if not gh_token or not github_repo:
        print("[git_push] GH_TOKEN / GH_REPO not set — skipping auto-push.")
        return

    print("[git_push] Auto-push enabled — committing and pushing output files...")

    try:
        _run(["git", "config", "user.email", "kaggle-bot@example.com"], cwd=repo_root)
        _run(["git", "config", "user.name", "Kaggle Auto Push"], cwd=repo_root)

        remote_url = f"https://{gh_token}@github.com/{github_repo}.git"
        _run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_root)

        rel_paths = [os.path.relpath(p, repo_root) for p in file_paths]

        status = subprocess.run(
            ["git", "status", "--porcelain"] + rel_paths, cwd=repo_root, capture_output=True, text=True
        )
        if not status.stdout.strip():
            print("[git_push] No changes to commit (outputs identical to last push).")
            return

        add_result = _run(["git", "add"] + rel_paths, cwd=repo_root)
        if add_result.returncode != 0:
            raise RuntimeError("git add failed")

        commit_result = _run(["git", "commit", "-m", commit_message], cwd=repo_root)
        if commit_result.returncode != 0:
            raise RuntimeError("git commit failed")

        push_result = _run(["git", "push"], cwd=repo_root)
        if push_result.returncode != 0:
            raise RuntimeError("git push failed")

        print("[git_push] Push complete.")
    except Exception as e:
        print(f"[git_push] WARNING: auto-push failed, continuing without it: {e}")
