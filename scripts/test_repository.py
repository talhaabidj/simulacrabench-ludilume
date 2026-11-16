"""Checks of the repository itself. They need no GPU, no model files and no
packages beyond pytest. Run: python3 -m pytest scripts"""

import ast
import os
import re

from fetch_weights import weight_url
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


def test_requirements_are_pinned_to_exact_versions():
    lines = (SUBMISSION / "requirements.txt").read_text().split()
    assert lines
    for line in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", line), line


def test_main_defines_predict_with_the_contract_arguments():
    tree = ast.parse((SUBMISSION / "main.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "predict" in functions
    assert [argument.arg for argument in functions["predict"].args.args] == ["frame", "schema"]


def test_model_files_are_fetched_at_full_commit_hashes():
    for name, record in load_manifest()["weights"].items():
        url = weight_url(name, record)
        assert re.fullmatch(r"https://huggingface\.co/[\w.-]+/[\w.-]+/resolve/[0-9a-f]{40}/[\w.-]+", url), url
