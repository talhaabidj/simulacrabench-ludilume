"""Read manifest.json and hash files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def load_manifest():
    return json.loads((ROOT / "manifest.json").read_text())


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, record):
    """True when the file exists with the recorded size and hash."""
    path = Path(path)
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and sha256_of(path) == record["sha256"])
