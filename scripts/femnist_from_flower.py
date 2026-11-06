"""Write the writers that datasets/femnist.yaml declares as LEAF JSON files.

The source is the Flower Datasets copy of FEMNIST (``flwrlabs/femnist`` on the
Hugging Face Hub: one Parquet file of LEAF's 28x28 images with their writer and
class). LEAF's own preprocessing (``leaf/data/femnist/preprocess.sh``) gives the
same layout from NIST SD19; this script is a shorter route to it, and writes only
the declared writers instead of the ~3,600.

The pixels follow LEAF's ``data_to_json.py``: grey levels divided by 255, so the
background is 1.0. Classes keep the Hub's order, which is LEAF's: 0-9 digits,
10-35 upper case, 36-61 lower case. Files hold 100 writers each, as LEAF's.

It needs pyarrow, Pillow and PyYAML, which are not dependencies of Onion-FL::

    uv run --no-project --with pyarrow --with pillow --with pyyaml --with numpy \\
        python -I scripts/femnist_from_flower.py \\
        data/FEMNIST/downloads/train-00000-of-00001.parquet data/FEMNIST/all_data
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

PER_FILE = 100  # LEAF's MAX_WRITERS


def declared_writers(descriptor: Path) -> list[str]:
    spec = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    return list(spec["source"]["writers"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--descriptor", type=Path, default=Path("datasets/femnist.yaml")
    )
    args = parser.parse_args()

    writers = declared_writers(args.descriptor)
    wanted = set(writers)
    data: dict[str, dict[str, list]] = {w: {"x": [], "y": []} for w in writers}
    table = pq.ParquetFile(args.parquet)
    for group in range(table.metadata.num_row_groups):
        rows = table.read_row_group(group, columns=["writer_id", "character", "image"])
        for writer, label, image in zip(
            *(rows.column(c).to_pylist() for c in ("writer_id", "character", "image")),
            strict=True,
        ):
            if writer not in wanted:
                continue
            pixels = np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("L"))
            if pixels.shape != (28, 28):
                raise SystemExit(f"{writer}: image of shape {pixels.shape}")
            data[writer]["x"].append((pixels.flatten() / 255).tolist())
            data[writer]["y"].append(int(label))
    missing = [w for w in writers if not data[w]["y"]]
    if missing:
        raise SystemExit(f"writers not in {args.parquet}: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(writers), PER_FILE):
        users = writers[k : k + PER_FILE]
        payload = {
            "users": users,
            "num_samples": [len(data[u]["y"]) for u in users],
            "user_data": {u: data[u] for u in users},
        }
        path = args.out / f"flower_{k // PER_FILE}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(path, len(users), sum(payload["num_samples"]))


if __name__ == "__main__":
    main()
