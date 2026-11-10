"""Write the real Nurse Stress extract to data/samples/NurseStress (issue #241).

Four whole E4 sessions of four nurses, each of them surveyed with a stress event
and time outside it, plus those nurses' rows of SurveyResults.xlsx, in the
layout of ``datasets/nurse_stress.yaml``, which reads it with ``root`` set to
the extract. Nothing is cut, resampled or invented: the files are copied (CC0,
Hosseini et al., 2022). tags.csv is left out, as the descriptor ignores it.

    python scripts/nurse_stress_sample.py [data/NurseStress] [data/samples/NurseStress]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

# One session per nurse, from April to December 2020 (2026-11-10).
SESSIONS = ["E4_1593538007", "F5_1595254546", "7E_1604522224", "BG_1607735684"]
FILES = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv", "HR.csv", "IBI.csv", "info.txt"]


def write_sample(source: Path, out: Path) -> None:
    for session in SESSIONS:
        nurse = session.split("_")[0]
        folder = out / "Stress_dataset" / nurse / session
        folder.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(
                source / "Stress_dataset" / nurse / session / name, folder / name
            )
    survey = pd.read_excel(source / "SurveyResults.xlsx", dtype={"ID": str})
    nurses = {s.split("_")[0] for s in SESSIONS}
    kept = survey[survey["ID"].isin(nurses)]
    kept.to_excel(out / "SurveyResults.xlsx", sheet_name="in", index=False)
    print(f"{out}: {len(SESSIONS)} sessions, {len(kept)} survey events")


if __name__ == "__main__":
    args = sys.argv[1:] + [None, None]
    write_sample(
        Path(args[0] or "data/NurseStress"), Path(args[1] or "data/samples/NurseStress")
    )
