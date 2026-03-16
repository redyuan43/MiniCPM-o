"""Host-specific path discovery helpers for local duplex tooling."""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Iterable


_CAPSWRITER_ROOT_ENV_KEYS = (
    "LOCAL_DUPLEX_CAPSWRITER_ROOT",
    "CAPSWRITER_ROOT",
)
_CAPSWRITER_PYTHON_ENV_KEYS = (
    "LOCAL_DUPLEX_CAPSWRITER_PYTHON",
    "CAPSWRITER_VENV_PYTHON",
)
_CAPSWRITER_REPO_NAMES = (
    "CapsWriter-Offline-Windows-64bit",
    "CapsWriter-Offline-Windows-64bit-main",
)


def _first_env_path(keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value).expanduser()
    return None


def _dedupe_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _candidate_home_dirs() -> list[Path]:
    homes = [
        Path.home(),
        Path(f"/home/{getpass.getuser()}"),
        Path("/home/dgx"),
        Path("/home/ivan"),
    ]
    return [path for path in _dedupe_paths(homes) if path.exists()]


def iter_capswriter_root_candidates() -> Iterable[Path]:
    env_python = _first_env_path(_CAPSWRITER_PYTHON_ENV_KEYS)
    env_root = _first_env_path(_CAPSWRITER_ROOT_ENV_KEYS)

    candidates: list[Path] = []
    if env_python is not None:
        candidates.append(env_python.parent.parent.parent)
    if env_root is not None:
        candidates.append(env_root)

    for home_dir in _candidate_home_dirs():
        github_dir = home_dir / "github"
        for repo_name in _CAPSWRITER_REPO_NAMES:
            candidates.append(github_dir / repo_name)
        if github_dir.exists():
            candidates.extend(sorted(github_dir.glob("CapsWriter-Offline-Windows-64bit*")))

    yield from _dedupe_paths(path.expanduser() for path in candidates)


def find_capswriter_root() -> Path | None:
    for candidate in iter_capswriter_root_candidates():
        if (candidate / "http_api_server.py").is_file():
            return candidate
    return None


def find_capswriter_python() -> Path | None:
    env_python = _first_env_path(_CAPSWRITER_PYTHON_ENV_KEYS)
    if env_python is not None and env_python.is_file():
        return env_python

    root = find_capswriter_root()
    if root is None:
        return None

    candidate = root / "venv-asr" / "bin" / "python"
    if candidate.is_file():
        return candidate
    return None


def require_capswriter_root() -> Path:
    root = find_capswriter_root()
    if root is not None:
        return root
    searched = ", ".join(str(path) for path in iter_capswriter_root_candidates())
    raise RuntimeError(
        "CapsWriter root not found. Set LOCAL_DUPLEX_CAPSWRITER_ROOT or install CapsWriter under one of: "
        f"{searched or 'no candidate directories'}"
    )


def require_capswriter_python() -> Path:
    python_path = find_capswriter_python()
    if python_path is not None:
        return python_path
    root = find_capswriter_root()
    if root is None:
        raise RuntimeError(
            "CapsWriter ASR Python not found because the CapsWriter root could not be resolved. "
            "Set LOCAL_DUPLEX_CAPSWRITER_PYTHON or LOCAL_DUPLEX_CAPSWRITER_ROOT."
        )
    raise RuntimeError(
        f"CapsWriter ASR Python not found under {root / 'venv-asr/bin/python'}. "
        "Set LOCAL_DUPLEX_CAPSWRITER_PYTHON to a valid interpreter."
    )
