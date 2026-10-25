"""Paths to data, checkpoints and outputs.

Values are read from experiments/paths.env, then from the environment.
Copy paths.example.env to paths.env and fill in your own paths.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

_env_file = HERE / "paths.env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip().removeprefix("export ").strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _value = _line.split("=", 1)
            _value = _value.split(" #", 1)[0].strip().strip("'\"")
            os.environ.setdefault(_key.strip(), os.path.expanduser(os.path.expandvars(_value)))

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", HERE / "outputs"))


def path(name):
    value = os.environ.get(name)
    if not value:
        raise KeyError(f"{name} is not set; add it to {_env_file}")
    return value


def optional_path(name):
    return os.environ.get(name) or None
