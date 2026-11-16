"""Checks of the repository itself. They need no GPU, no model files and no
packages beyond pytest. Run: python3 -m pytest scripts"""

import os

from manifest import SUBMISSION, load_manifest, matches

IGNORED_NAMES = {".DS_Store"}
IGNORED_FOLDERS = {"__pycache__"}


def files_on_disk():
    """Relative paths of the files under submission/, model files included."""
    found = set()
    for folder, subfolders, names in os.walk(SUBMISSION):
        subfolders[:] = [name for name in subfolders if name not in IGNORED_FOLDERS]
        for name in names:
            if name not in IGNORED_NAMES and not name.endswith(".part"):
                found.add((SUBMISSION / folder / name).relative_to(SUBMISSION).as_posix())
    return found


def test_every_committed_file_matches_the_manifest():
    manifest = load_manifest()
    wrong = [name for name, record in manifest["files"].items()
             if not matches(SUBMISSION / name, record)]
    assert wrong == []


def test_submission_folder_holds_nothing_unlisted():
    manifest = load_manifest()
    listed = set(manifest["files"]) | set(manifest["weights"])
    assert files_on_disk() - listed == set()
