"""Rebuild the Test submission zip and compare it with the one that was scored.

    python3 scripts/build_submission.py

Every file in submission/ is checked against manifest.json first. The zip is
written with fixed names, order, dates and permissions, so the same Python and
zlib give the same bytes. The file hashes are the guarantee; the zip hash also
depends on the zlib version (the submitted zip was built with zlib 1.2.12).
"""

import sys
import zipfile

from manifest import ROOT, SUBMISSION, load_manifest, matches, sha256_of

ZIP_PATH = ROOT / "dist" / "submission.zip"
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644 << 16


def check_files(records):
    """Names of files that are missing or differ from the manifest."""
    return [name for name, record in records.items() if not matches(SUBMISSION / name, record)]


def write_zip(names, folder=SUBMISSION, zip_path=ZIP_PATH):
    """Write the named files from folder into zip_path, the same way every time."""
    zip_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = FILE_MODE
            with archive.open(info, "w") as entry, open(folder / name, "rb") as source:
                for block in iter(lambda: source.read(1 << 22), b""):
                    entry.write(block)


def main():
    manifest = load_manifest()
    records = {**manifest["files"], **manifest["weights"]}
    problems = check_files(records)
    if problems:
        print("These files are missing or differ from manifest.json:")
        for name in problems:
            print("  " + name)
        print("Run scripts/fetch_weights.py for the model files.")
        return 1
    print(f"{len(records)} files match manifest.json")
    write_zip(records)
    built = sha256_of(ZIP_PATH)
    print(ZIP_PATH)
    print("SHA-256  ", built)
    print("submitted", manifest["submission"]["zip_sha256"])
    if built == manifest["submission"]["zip_sha256"]:
        print("The zip is byte-identical to the Test submission.")
    else:
        print("Same files, different zip bytes (another zlib version). The contents are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
