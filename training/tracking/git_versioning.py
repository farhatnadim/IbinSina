"""Git versioning utilities for experiment reproducibility."""

import subprocess
from datetime import datetime
from typing import Dict, Optional


class GitVersioningError(Exception):
    """Raised when git state prevents experiment tracking."""

    pass


def _run_git(*args: str, check: bool = True) -> str:
    """Run git command and return output.

    Args:
        *args: Git command arguments
        check: If True, raise on non-zero exit code

    Returns:
        Command stdout stripped of whitespace

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
        GitVersioningError: If git is not available
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise GitVersioningError("Git is not installed or not in PATH")
    except subprocess.CalledProcessError as e:
        if check:
            raise GitVersioningError(f"Git command failed: {e.stderr.strip()}")
        return ""


def is_git_repo() -> bool:
    """Check if current directory is inside a git repository."""
    try:
        _run_git("rev-parse", "--git-dir")
        return True
    except GitVersioningError:
        return False


def has_uncommitted_changes() -> bool:
    """Check if repo has uncommitted changes.

    Returns:
        True if there are staged or unstaged changes
    """
    if not is_git_repo():
        return False

    status = _run_git("status", "--porcelain")
    return len(status.strip()) > 0


def get_git_info() -> Dict[str, str]:
    """Get current git commit, branch, dirty status.

    Returns:
        Dictionary with git information:
        - git_commit: Full commit hash
        - git_commit_short: Short commit hash (7 chars)
        - git_branch: Current branch name
        - git_dirty: "True" or "False" string
    """
    if not is_git_repo():
        return {
            "git_commit": "not_a_repo",
            "git_commit_short": "not_a_repo",
            "git_branch": "not_a_repo",
            "git_dirty": "False",
        }

    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_commit_short": _run_git("rev-parse", "--short", "HEAD"),
        "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(has_uncommitted_changes()),
    }


def ensure_clean_repo() -> None:
    """Raise error if repo has uncommitted changes.

    This enforces strict reproducibility by requiring all code changes
    to be committed before running experiments.

    Raises:
        GitVersioningError: If repository has uncommitted changes
    """
    if not is_git_repo():
        raise GitVersioningError(
            "Not inside a git repository. "
            "Please initialize a git repo for experiment tracking."
        )

    if has_uncommitted_changes():
        raise GitVersioningError(
            "Git repository has uncommitted changes. "
            "Please commit or stash changes before running experiments "
            "to ensure reproducibility. You can disable this check by "
            "setting git_tag: false in your tracking config."
        )


def create_experiment_tag(
    run_name: str,
    metrics: Dict[str, float],
    push: bool = False,
) -> str:
    """Create annotated git tag for experiment.

    Tag format: exp/<run_name>/<timestamp>
    Tag message includes metrics for quick reference.

    Args:
        run_name: Experiment run name
        metrics: Final metrics dict (accuracy, kappa, etc.)
        push: Whether to push tag to remote

    Returns:
        Tag name that was created

    Raises:
        GitVersioningError: If tagging fails
    """
    if not is_git_repo():
        raise GitVersioningError("Not inside a git repository")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize run_name for git tag (replace invalid characters)
    safe_run_name = run_name.replace(" ", "_").replace("/", "-").replace(":", "-")
    tag_name = f"exp/{safe_run_name}/{timestamp}"

    # Build annotated tag message with metrics
    message_lines = [
        f"Experiment: {run_name}",
        f"Timestamp: {timestamp}",
        "",
        "Metrics:",
    ]
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            message_lines.append(f"  {key}: {value:.4f}")
        else:
            message_lines.append(f"  {key}: {value}")

    message = "\n".join(message_lines)

    # Create annotated tag
    try:
        _run_git("tag", "-a", tag_name, "-m", message)
        print(f"Created git tag: {tag_name}")
    except GitVersioningError as e:
        raise GitVersioningError(f"Failed to create tag '{tag_name}': {e}")

    if push:
        try:
            _run_git("push", "origin", tag_name)
            print(f"Pushed git tag to origin: {tag_name}")
        except GitVersioningError as e:
            print(f"Warning: Failed to push tag to remote: {e}")
            # Don't raise - tag was created locally which is the important part

    return tag_name


def delete_experiment_tag(tag_name: str, remote: bool = False) -> None:
    """Delete an experiment tag.

    Args:
        tag_name: Tag name to delete
        remote: If True, also delete from remote

    Raises:
        GitVersioningError: If deletion fails
    """
    if not is_git_repo():
        raise GitVersioningError("Not inside a git repository")

    # Delete local tag
    _run_git("tag", "-d", tag_name)
    print(f"Deleted local tag: {tag_name}")

    if remote:
        try:
            _run_git("push", "origin", f":refs/tags/{tag_name}")
            print(f"Deleted remote tag: {tag_name}")
        except GitVersioningError as e:
            print(f"Warning: Failed to delete remote tag: {e}")


def list_experiment_tags(pattern: Optional[str] = None) -> list:
    """List experiment tags.

    Args:
        pattern: Optional glob pattern to filter tags (e.g., "exp/panda_*")

    Returns:
        List of tag names matching pattern
    """
    if not is_git_repo():
        return []

    if pattern:
        output = _run_git("tag", "-l", pattern)
    else:
        output = _run_git("tag", "-l", "exp/*")

    if not output:
        return []

    return output.split("\n")
