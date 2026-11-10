"""Unpack Nurse Stress's Stress_dataset.zip and its nested session zips.

The download (Dryad doi:10.5061/dryad.5hqbzkh6f, or Zenodo record 5514277) holds
``<ID>/<ID>_<unix start>.zip``, one zip per E4 session. This writes each session
to ``<out>/<ID>/<ID>_<unix start>/`` (3.6 GB for the 609 sessions), the layout
``datasets/nurse_stress.yaml`` reads. The archive is untrusted input: every
member name is checked, only the expected CSV and text files are written, and
nothing leaves ``<out>``. Run it with the standard library only:

    python -I scripts/nurse_stress_extract.py \
        data/NurseStress/Stress_dataset.zip data/NurseStress/Stress_dataset
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

SESSION = re.compile(r"^([A-Za-z0-9]{2})/\1_(\d{10})\.zip$")
MEMBER = re.compile(r"^[A-Za-z0-9_]+\.(csv|txt)$")
LIMIT = 2_000_000_000  # bytes per member, far above the largest ACC.csv


def _inside(base: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or ":" in name or "\\" in name:
        raise ValueError(f"unsafe member name {name!r}")
    target = (base / pure).resolve()
    if base not in target.parents:
        raise ValueError(f"{name!r} would leave {base}")
    return target


def extract(archive: Path, out: Path) -> int:
    out = out.resolve()
    sessions = 0
    with zipfile.ZipFile(archive) as outer:
        for info in outer.infolist():
            if info.is_dir():
                continue
            match = SESSION.match(info.filename)
            if match is None:
                raise ValueError(f"unexpected file {info.filename!r} in {archive}")
            nurse, start = match.groups()
            folder = _inside(out, f"{nurse}/{nurse}_{start}")
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(outer.read(info))) as inner:
                for member in inner.infolist():
                    if member.is_dir():
                        continue
                    if not MEMBER.match(member.filename) or member.file_size > LIMIT:
                        raise ValueError(f"unexpected {member.filename!r} in {info}")
                    target = _inside(folder, member.filename)
                    target.write_bytes(inner.read(member))
            sessions += 1
    return sessions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(f"{extract(Path(sys.argv[1]), Path(sys.argv[2]))} sessions")
