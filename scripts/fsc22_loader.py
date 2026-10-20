"""Loads FSC22 clips mapped to the eco-sentry class taxonomy.

FSC22 (Forest Sound Classification Dataset) is forest-specific and, unlike
ESC-50, it actually contains both a Chainsaw category AND a Gunshot
category -- the two classes ESC-50 and UrbanSound8K cannot cover alone.
This is why FSC22 is pulled in: ESC-50's ~40 chainsaw clips were the
weakest link in the real-audio evaluation (recall ~0.54), and FSC22
provides 75 labelled chainsaw clips to fix exactly that gap.

Mapping rules follow the same strict discipline as the other two loaders:
only exact category strings from the FSC22 metadata CSV's "Class Name"
column are keys here, and every mapped-to class string must be an exact
key of ecosentry.config.CLASS_MAP.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ecosentry.arch1_audio import _read_wav
from ecosentry.config import CLASS_MAP
from ecosentry.synth import SynthSample

#: FSC22 "Class Name" column value -> eco-sentry class name. Every key
#: here must be an exact string from "Metadata V1.0 FSC22.csv"'s
#: "Class Name" column (verified: Chainsaw 75, Gunshot 75).
#:
#: Deliberately kept minimal -- only the two classes neither ESC-50 nor
#: UrbanSound8K covers (chainsaw: only ESC-50's ~40 clips; gunshot: only
#: US8K's gun_shot). Earlier experiments showed pulling in FSC22's
#: VehicleEngine/Generator/Fire/etc. actively HURT overall accuracy
#: (71.3% vs 83.6%) because freesound's recording conditions and loose
#: labelling conflict with ESC-50/US8K's cleaner notion of vehicle/ambient.
CATEGORY_TO_CLASS: Dict[str, str] = {
    "Chainsaw": "chainsaw",
    "Gunshot": "gunshot",
}

#: Every other FSC22 category (Fire, Rain, Wind, VehicleEngine, Axe,
#: Handsaw, WoodChop, Generator, Helicopter, Footsteps, Speaking, ...) is
#: deliberately excluded: experiment 4 showed loading them all reduced
#: real-audio accuracy from 83.6% to 71.3%, so only the two gap-filling
#: classes are kept here.


def load_fsc22_corpus(
    root: Path, max_per_category: Optional[int] = None
) -> Tuple[List[SynthSample], Dict[str, int], List[Tuple[str, str]]]:
    """Load every FSC22 clip whose class is in CATEGORY_TO_CLASS.

    root must be the FSC22 repository root (the directory containing
    `Audios/` and `Metadata/`).

    Returns (samples, counts, skipped) where skipped is a list of
    (filename, reason) pairs for any file that failed to read -- expected
    to happen occasionally on a real-world dataset this size and handled
    by skipping, not crashing, but every skip is reported so nothing
    silently vanishes from the final counts.
    """
    root = Path(root)
    meta_path = root / "Metadata" / "Metadata V1.0 FSC22.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No FSC22 metadata CSV found at {meta_path}. "
        )

    with open(meta_path) as f:
        rows = list(csv.DictReader(f))

    samples: List[SynthSample] = []
    counts: Dict[str, int] = {}
    skipped: List[Tuple[str, str]] = []
    for row in rows:
        category = row["Class Name"]
        if category not in CATEGORY_TO_CLASS:
            continue
        if max_per_category is not None and counts.get(category, 0) >= max_per_category:
            continue
        wav_path = root / "Audios" / row["Dataset File Name"]
        try:
            audio, sr = _read_wav(wav_path)
        except Exception as exc:  # real-world datasets have occasional bad files
            skipped.append((row["Dataset File Name"], f"{type(exc).__name__}: {exc}"))
            continue
        our_class = CATEGORY_TO_CLASS[category]
        samples.append(
            SynthSample(
                audio=audio,
                label=CLASS_MAP[our_class],
                forest="mixed",
                source=f"FSC22:{category}:id{row['Class ID']}",
                sample_rate=sr,
            )
        )
        counts[category] = counts.get(category, 0) + 1
    return samples, counts, skipped
