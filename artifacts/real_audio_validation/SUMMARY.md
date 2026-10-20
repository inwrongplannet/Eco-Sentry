# Real-Audio Validation Summary

## 1. Setup

**Datasets used** (all real field recordings, none synthetic):

| Dataset | Clips | Notes |
|---|---|---|
| ESC-50 | 2,000 (480 mapped) | 5 s environmental clips; no `gunshot` category |
| UrbanSound8K | 8,732 (4,732 mapped) | urban clips; provides the `gunshot` class |
| FSC22 (Forest Sound Classification) | 2,025 | forest clips, 27 classes incl. `Chainsaw` (75) and `Gunshot` (75) |

**Taxonomy (4 incident classes):** `gunshot`, `chainsaw`, `vehicle`, `ambient`.

**Class mappings (source category -> eco-sentry class):**
- ESC-50: `chainsaw` -> chainsaw; `engine`, `car_horn`, `train`, `airplane`, `helicopter` -> vehicle; `wind`, `rain`, `crickets`, `chirping_birds`, `crackling_fire`, `thunderstorm` -> ambient. **No gunshot.**
- UrbanSound8K: `gun_shot` -> gunshot; `car_horn`, `engine_idling`, `siren` -> vehicle; `air_conditioner`, `street_music` -> ambient. **No chainsaw.**
- FSC22: `Chainsaw` -> chainsaw; `Gunshot` -> gunshot (minimal mapping only -- see section 6).

**Important caveat on splits:** the retraining split is a random stratified split seeded by `DatasetConfig.seed` (fixed at 42). Within a multi-seed run the *same* real test set is reused, so seed-to-seed variation isolates training/initialization randomness (not split differences). Cross-config comparisons differ in both budget and split, so they are not directly comparable.

## 2. Experiment 1: Zero-Shot Transfer (synthetic -> ESC-50)

A model trained *entirely on synthetic audio* evaluated on real ESC-50 audio it has never seen:

- **Argmax accuracy: 16.0%** (n=480)
- Alert-decision rates by true class: `chainsaw` 42.5% (n=40), `vehicle` 34.5% (n=200), `ambient` 18.8% (n=240)

Gunshot is not evaluated here: ESC-50 contains no `gun_shot` category (confirmed against `meta/esc50.csv`).

## 3. Experiment 2: Retraining with Real ESC-50 Only

Retrain the model with real ESC-50 audio mixed into the synthetic corpus:

- **Real-only test accuracy: 65.8%** (n=146)
- Combined (synthetic + real) test accuracy: 75.5%
- Val accuracy: 80.6%

(Historical 5-seed run: 67.7% +/- 2.7%.)

## 4. Experiment 3: UrbanSound8K Zero-Shot (gunshot)

Zero-shot evaluation of the synthetic-only model on mapped UrbanSound8K clips:

- Overall argmax accuracy: 7.8% (n=4,732)
- **Gunshot alert rate: 9.4%** (n=374)
- Vehicle alert rate 39.0% (n=2,358); ambient 43.5% (n=2,000)

This establishes that a purely synthetic-trained model cannot detect real gunshots; real data is required.

## 5. Combined Retraining (ESC-50 + UrbanSound8K), Full Budget

Config: 1,200 synthetic + 480 ESC-50 + 2,374 UrbanSound8K clips (cap 400/category), 120 epochs, 5 seeds (42-46).

**Result: 80.2% +/- 4.0% real-only accuracy (n=671 real clips in the fixed test split).**
**Gunshot recall: 81.0% +/- 9.2%.**

| Seed | Real-only acc | Gunshot recall | n (real test) |
|---|---|---|---|
| 42 | 83.6% | 88.1% | 671 |
| 43 | 77.4% | 72.9% | 671 |
| 44 | 83.7% | 91.5% | 671 |
| 45 | 74.8% | 70.4% | 671 |
| 46 | 81.6% | 82.0% | 671 |

Full per-seed JSON is in `iterations/`.

### Data-budget effect (important)
An earlier **smaller-budget** run (600 synthetic, UrbanSound8K capped at 60/category -> 360 clips, 60 epochs, 218 real test) over 10 seeds gave **65.8% +/- 3.5%**. Increasing the real-data budget to the full mapped corpus raises the result to **80.2% +/- 4.0%**. The archived small-budget run is preserved in `iterations_small_budget_10seed/`. The improvement is driven by the amount of real training audio, not by any architectural change.

## 6. Adding FSC22 (Forest Dataset)

FSC22 was added to supply real `chainsaw` and forest-domain audio, which ESC-50 and UrbanSound8K lack.

- **Full FSC22 mapping (all 27 classes):** 71.3% overall. Much of FSC22 is free-field field recordings of non-incident sounds (rain, insects, birds, etc.) whose acoustic character differs from the urban/ESC-50 pool; adding all of it degrades the aggregate.
- **Minimal FSC22 mapping (Chainsaw + Gunshot only):** 77.9% overall, **gunshot recall 91.5%**, **chainsaw recall 60.5%**. FSC22 clearly helps the two under-covered incident classes, but its noisy clips slightly reduce the aggregate relative to ESC-50+US8K.

## 7. Honest Ceiling

On this 4-class incident taxonomy the model plateaus in the **~80-84%** real-only range (best single seed 83.6%; full-budget 5-seed mean 80.2%). **Real-audio accuracy above 90% was not achieved.**

Contributing factors:
- The ~16k-parameter SNN and the coarse 4-class taxonomy integrate very different acoustic sources (e.g. `vehicle` = engine + car horn + train + airplane + helicopter + siren).
- Domain mismatch: `ambient` is validated on *urban* clips (air conditioner, street music) rather than the forest ambience the class targets.
- `chainsaw` has very small real support in the combined ESC-50+US8K pool (no real chainsaw); only FSC22 provides it.
- Reaching >90% would require architectural/taxonomy changes (out of scope here) or substantially more curated real data per incident class.

## 8. Limitations
- **Gunshot** evaluated with real audio only via UrbanSound8K; **chainsaw** only via FSC22.
- **Ambient** validated on urban ambience (semantic mismatch vs forest target).
- **Non-standard folds:** UrbanSound8K's official 10-fold protocol is not used (random stratified splits instead), preventing direct comparison with published UrbanSound8K benchmarks.
- **Clip length:** 5 s source clips padded to the model's 10 s window.
- Results are single-machine CPU runs; no GPU.

Regenerate with:
`scripts/run_experiment1_zero_shot.py`, `scripts/run_experiment2_retrain_with_real.py`, `scripts/run_experiment3_us8k_zero_shot.py`, and `scripts/run_iterations_combined_datasets.py` (see `--esc50-root`/`--us8k-root`/`--fsc22-root`).
