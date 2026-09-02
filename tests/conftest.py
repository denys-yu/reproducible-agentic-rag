"""Shared fixtures.

`assert_output_dir_not_ignored` fails CLOSED: git must be able to answer "is this ignored?", and
outside a work tree it cannot, so it raises. A bare `tmp_path` is outside any work tree, which
would make every driver test trip the gate for a reason unrelated to what it is testing. The
fixture below turns `tmp_path` into a real (empty) repository, so the gate runs for real and
answers "not ignored" — exercising the production path rather than bypassing it.
"""

from __future__ import annotations

import subprocess

import pytest


def init_git_repo(path):
    """Initialise an empty git repo at `path` so `git check-ignore` can answer about it."""
    subprocess.run(
        ["git", "init", "--quiet", str(path)],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def git_tmp_path(tmp_path):
    """A `tmp_path` that is a git work tree, for tests that write into a checked output dir."""
    return init_git_repo(tmp_path)
