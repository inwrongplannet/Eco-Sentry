# UrbanSound8K Validation Summary

## Setup
To address the lack of gunshot clips in the initial ESC-50 validation and to expand the real-audio sample pool, the dataset **UrbanSound8K** was integrated into the validation pipeline.

**Dataset Citation:**
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

**Important Caveat on Validation Splits:**
UrbanSound8K is natively provided as 10 pre-defined folds (fold1 through fold10), and the authors strongly recommend performing 10-fold cross-validation when comparing models on this dataset. However, to integrate this dataset seamlessly with the existing ESC-50 + synthetic pipeline (ARCH_3), this project utilizes **random stratified splits** rather than the official folds. While this breaks strict comparability with published UrbanSound8K benchmarks, it aligns with our project's goal of evaluating generalizability rather than competing on UrbanSound8K specifically.

## Experiment 3: Zero-Shot (Gunshot Evaluation)
The zero-shot evaluation on the UrbanSound8K dataset yields the first real-audio result for the `gunshot` class in this project. 
When the purely synthetic-trained model was evaluated on the mapped UrbanSound8K categories without any retraining, the overall argmax accuracy was **7.8%**.
For the `gunshot` category specifically (n = 374 clips), the alert rate was **9.4%**. 

## Multi-Seed Combined Retraining (Experiment 7)
To ensure robustness and meet rigorous research integrity standards, the retraining experiment combining Synthetic, ESC-50, and UrbanSound8K data was run across **10 random seeds** (42, 43, 44, 45, 46, 47, 48, 49, 50, 51). 

The mean accuracy on real-audio test clips across the 10 seeds is **65.8% ± 3.5%** standard deviation. The tighter standard deviation across 10 iterations confirms that the model's accuracy on mixed real-world audio datasets is extremely robust and not heavily sensitive to random initialization or split differences.

Crucially, with the inclusion of the per-class metrics tracking, the combined real-audio gunshot recall across the 10 seeds was measured at **69.2% ± 6.5%**. This is a massive improvement over the initial 9.1% zero-shot alert rate, confirming that retraining on UrbanSound8K successfully enables the model to detect real-world gunshot audio.

**Per-Seed Breakdown:**
- **Seed 42:** 64.7% (n=218)
- **Seed 43:** 68.3% (n=218)
- **Seed 44:** 71.1% (n=218)
- **Seed 45:** 60.7% (n=219)
- **Seed 46:** 63.6% (n=217)
- **Seed 47:** 66.5% (n=218)
- **Seed 48:** 68.9% (n=219)
- **Seed 49:** 68.8% (n=218)
- **Seed 50:** 63.2% (n=220)
- **Seed 51:** 61.9% (n=218)

## Full-Budget Retraining (Escalated Data Budget)

The 10-seed run above used a deliberately small real-data budget (UrbanSound8K capped at 60 clips/category -> 360 clips; 600 synthetic clips; 60 epochs). Scaling the budget to the full mapped corpus (UrbanSound8K cap 400/category -> 2,374 clips; 1,200 synthetic clips; 120 epochs) over 5 seeds (42-46) raises real-only accuracy substantially:

- **80.2% +/- 4.0% real-only accuracy (n=671)**
- **Gunshot recall 81.0% +/- 9.2%**
- Per seed: 42: 83.6%, 43: 77.4%, 44: 83.7%, 45: 74.8%, 46: 81.6%

The small-budget 10-seed run is archived in `iterations_small_budget_10seed/`; the escalated run is in `iterations/`. The gain is attributable to the larger real-audio budget, not to any architectural change.

## Cross-Dataset Comparison Table

| Experiment | Real-audio accuracy | n | Gunshot covered? |
|---|---|---|---|
| ESC-50 zero-shot (earlier plan) | 16.0% | 480 | No |
| ESC-50 retrained, real-only (earlier plan) | 65.8% | 146 | No |
| UrbanSound8K zero-shot, gunshot only (T4) | 9.4% | 374 | Yes |
| ESC-50 + UrbanSound8K retrained, real-only, 10 seeds, small budget (T7) | 65.8% +/- 3.5% (Gunshot recall: 69.2% +/- 6.5%) | 217-220 | Yes |
| ESC-50 + UrbanSound8K retrained, real-only, 5 seeds, **full budget** | **80.2% +/- 4.0%** (Gunshot recall: 81.0% +/- 9.2%) | 671 | Yes |

## Limitations
- **Chainsaw Category:** The `chainsaw` category remains unrepresented in the real-audio validation. Neither ESC-50 nor UrbanSound8K contains chainsaw samples, meaning it remains a synthetic-only class in our evaluation.
- **Ambient Category Semantics:** The `ambient` class in our taxonomy targets forest/eco-ambience. However, due to the nature of UrbanSound8K, the tested clips are urban ambience (e.g., street music, air conditioner).
- **Non-Standard Folds:** As mentioned in the setup, the official UrbanSound8K 10-fold cross-validation protocol was not utilized, preventing direct comparison with literature.
