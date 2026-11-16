"""Checks for the manifest helpers. Run: python3 -m pytest scripts"""

import hashlib

from manifest import load_manifest, matches, sha256_of


def test_matches_needs_right_size_and_hash(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    record = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert sha256_of(path) == record["sha256"]
    assert matches(path, record)
    assert not matches(path, dict(record, bytes=4))
    assert not matches(path, dict(record, sha256="0" * 64))
    assert not matches(tmp_path / "missing.bin", record)


def test_manifest_lists_the_three_models_and_the_scored_zip():
    manifest = load_manifest()
    assert len(manifest["weights"]) == 3
    assert len(manifest["files"]) + len(manifest["weights"]) == 156
    assert len(manifest["submission"]["zip_sha256"]) == 64
    for record in manifest["weights"].values():
        assert len(record["commit"]) == 40 and len(record["sha256"]) == 64
