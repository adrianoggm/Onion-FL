from __future__ import annotations

"""Copy a real extract of Stress-Predict into ``data/samples/stress_predict/``.

Three whole E4 sessions (S02, S03, S04, about 8 MB), the time log and the MIT
licence, laid out like the dataset so ``datasets/stress_predict.yaml`` reads it
with ``root`` changed. Nothing is generated: the files are copied as they are.
Stress-Predict is MIT-licensed, which allows the copy with its notice, so the
licence goes with it, plus the citation the authors ask for.

    python scripts/stress_predict_sample.py [data/StressPredict] [data/samples/stress_predict]
"""

import shutil
import sys
from pathlib import Path

SUBJECTS = ["S02", "S03", "S04"]
SIGNALS = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]
ATTRIBUTION = """\
# Stress-Predict extract

A copy of three sessions (S02, S03, S04) and Processed_data/Time_logs.xlsx of
the Stress-Predict Dataset, https://github.com/italha-d/Stress-Predict-Dataset
(commit f49d2ea), under the MIT licence in LICENSE. Copyright (c) 2022 Talha
Iqbal. The files are unchanged.

Cite: T. Iqbal et al., "Stress Monitoring Using Wearable Sensors: A Pilot Study
and Stress-Predict Dataset", Sensors 22(21):8135, 2022,
https://doi.org/10.3390/s22218135.

Written by scripts/stress_predict_sample.py of Onion-FL.
"""


def main(source: Path, target: Path) -> None:
    for subject in SUBJECTS:
        folder = target / "Raw_data" / subject
        folder.mkdir(parents=True, exist_ok=True)
        for signal in SIGNALS:
            shutil.copyfile(
                source / "Raw_data" / subject / f"{signal}.csv",
                folder / f"{signal}.csv",
            )
    (target / "Processed_data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        source / "Processed_data" / "Time_logs.xlsx",
        target / "Processed_data" / "Time_logs.xlsx",
    )
    shutil.copyfile(source / "LICENSE", target / "LICENSE")
    (target / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    args = sys.argv[1:] + [None, None]
    main(
        Path(args[0] or "data/StressPredict"),
        Path(args[1] or "data/samples/stress_predict"),
    )
