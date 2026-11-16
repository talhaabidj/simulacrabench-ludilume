"""Download the three pretrained model files at their pinned commits.

    python3 scripts/fetch_weights.py

Each file comes from Hugging Face at the commit named in manifest.json and is
kept only if its size and SHA-256 match. A file that is already in place and
matches is not downloaded again. A download that is cut off continues from
where it stopped, in the same run or the next one.
"""

import http.client
import shutil
import sys
import time
import urllib.request

from manifest import SUBMISSION, load_manifest, matches

ATTEMPTS = 8
WAIT_SECONDS = 5


def weight_url(name, record):
    """Address of a model file at its pinned commit."""
    return f"https://huggingface.co/{record['repo']}/resolve/{record['commit']}/{name}"


def size_of(path):
    return path.stat().st_size if path.exists() else 0


def fetch_rest(url, partial):
    """Append to the partial file what it still lacks of url."""
    done = size_of(partial)
    headers = {"Range": f"bytes={done}-"} if done else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        # 206 means the server sent only the missing part; otherwise start again.
        with open(partial, "ab" if response.status == 206 else "wb") as output:
            shutil.copyfileobj(response, output, length=1 << 22)


def download(url, target, size, attempts=ATTEMPTS, wait=WAIT_SECONDS):
    """Download url next to target as a .part file of the given size.

    A connection that drops, with or without an error, leaves a short file;
    the next attempt asks only for the rest.
    """
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            fetch_rest(url, partial)
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            print(f"         connection dropped ({error})")
        if size_of(partial) >= size:
            break
        if attempt < attempts:
            print(f"         {size_of(partial) / 1e6:.0f} of {size / 1e6:.0f} MB so far; continuing in {wait} s")
            time.sleep(wait)
    return partial


def main():
    failed = []
    for name, record in load_manifest()["weights"].items():
        target = SUBMISSION / name
        if matches(target, record):
            print(f"ok       {name}")
            continue
        print(f"download {name} ({record['bytes'] / 1e6:.0f} MB)")
        partial = download(weight_url(name, record), target, record["bytes"])
        if matches(partial, record):
            partial.replace(target)
            print(f"ok       {name}")
        elif size_of(partial) < record["bytes"]:
            failed.append(name)
            print(f"INCOMPLETE {name}: run this script again to continue the download")
        else:
            partial.unlink()
            failed.append(name)
            print(f"MISMATCH {name}: size or SHA-256 differs from manifest.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
