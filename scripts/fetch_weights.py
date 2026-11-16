"""Download the three pretrained model files at their pinned commits.

    python3 scripts/fetch_weights.py

Each file comes from Hugging Face at the commit named in manifest.json and is
kept only if its size and SHA-256 match. A file that is already in place and
matches is not downloaded again.
"""

import shutil
import sys
import urllib.request

from manifest import SUBMISSION, load_manifest, matches


def weight_url(name, record):
    """Address of a model file at its pinned commit."""
    return f"https://huggingface.co/{record['repo']}/resolve/{record['commit']}/{name}"


def download(url, target):
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as output:
        shutil.copyfileobj(response, output, length=1 << 22)
    return partial


def main():
    failed = []
    for name, record in load_manifest()["weights"].items():
        target = SUBMISSION / name
        if matches(target, record):
            print(f"ok       {name}")
            continue
        print(f"download {name} ({record['bytes'] / 1e6:.0f} MB)")
        partial = download(weight_url(name, record), target)
        if matches(partial, record):
            partial.replace(target)
            print(f"ok       {name}")
        else:
            partial.unlink()
            failed.append(name)
            print(f"MISMATCH {name}: size or SHA-256 differs from manifest.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
