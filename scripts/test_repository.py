"""Checks of the repository itself. They need no GPU, no model files and no
packages beyond pytest. Run: python3 -m pytest scripts"""

import ast
import os
import re
import zipfile

from build_submission import FIXED_DATE, write_zip
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


def test_zip_bytes_do_not_depend_on_file_times_or_listing_order(tmp_path):
    folder = tmp_path / "files"
    (folder / "sub").mkdir(parents=True)
    (folder / "b.txt").write_bytes(b"second")
    (folder / "sub" / "a.txt").write_bytes(b"first" * 1000)
    os.utime(folder / "b.txt", (1_600_000_000, 1_600_000_000))

    write_zip(["sub/a.txt", "b.txt"], folder, tmp_path / "one.zip")
    os.utime(folder / "b.txt", (1_700_000_000, 1_700_000_000))
    write_zip(["b.txt", "sub/a.txt"], folder, tmp_path / "two.zip")

    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "one.zip") as archive:
        assert archive.namelist() == ["b.txt", "sub/a.txt"]
        assert {entry.date_time for entry in archive.infolist()} == {FIXED_DATE}
