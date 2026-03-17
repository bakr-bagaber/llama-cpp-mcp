"""Test helpers for the src-layout package."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_WORK = ROOT / "test-workspace"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sandbox_path() -> Path:
    """Create a scratch directory inside the workspace.

    Using a workspace-local directory avoids the Windows temp cleanup
    permission issues we hit with pytest's built-in tmp_path fixture.
    """
    TEST_WORK.mkdir(exist_ok=True)
    path = TEST_WORK / uuid.uuid4().hex
    path.mkdir()
    return path


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only.

    The project does not currently depend on trio, so locking the backend
    keeps the runtime tests deterministic.
    """
    return "asyncio"
