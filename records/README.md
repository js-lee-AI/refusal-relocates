# Run records

Per-run numbers behind the paper's tables, one row per run and arm. They hold rates, spectral statistics and losses only. No prompt, generation, training example or adapter weight is stored here. The scripts in `experiments/` read these files and print the paper's tables.

Rates are fractions of the evaluation prompts. `refusal` is coherent refusal on AdvBench, `lg_unsafe` the Llama-Guard-3 unsafe rate on AdvBench and `over_refusal` coherent refusal on the XSTest safe prompts. `gate` is 1 when the run passes the validity gate of Appendix B. Seeds are 42, 43 and 44. `frozen_upto` is the last frozen layer b of a `[0, b]` freeze, and empty for an unrestricted attack.

| file | rows | used by |
|---|---|---|
| `freeze.csv` | one per run and arm (`clean`, `attack`, `freeze`, `matched`) | Tables 1, 2, 7 and 12 |
| `lockstep.csv` | one per run and patched layer | Tables 8, 9 and 10, Figure 5 |
| `topk.csv` | one per attacked adapter | Table 16 |
| `adaptive.csv` | one per Llama-3.1-8B dose-100 adapter | Tables 3, 17 and 19, the detector |
| `benign_spectra.csv` | one per benign fine-tune | the detector |
| `mixed_freeze.csv` | one per run and arm | Table 14 |
| `rank_ladder.csv` | one per fine-tune and arm | Table 20 |

## freeze.csv

| column | meaning |
|---|---|
| `dose` | harmful training examples |
| `arm` | `clean` model, unrestricted `attack`, band `freeze`, or the `matched` freeze of equal size at the other end |
| `n_eval` | AdvBench prompts scored |

## lockstep.csv

| column | meaning |
|---|---|
| `study` | `dose` (the sweep of Table 8), `band` (unrestricted and band-frozen adapters on the coarse grid), `ladder` (Table 9) or `step_one` (Table 10) |
| `arm` | `floor` (attacked model alone), `ceiling` (clean model alone), `self_patch` (the attacked model patched with its own state at `layer`), `positive_control` (the clean state patched in at the last layer), or `patch` (the clean state patched in at `layer`) |
| `refusal` | coherent refusal of the lockstep generation |

## topk.csv

`clean` and `attacked` are refusal before any repair. `rmtop1`, `rmtop2` and `rmtop4` are refusal after removing the top one, two or four singular directions of every adapted module, and `rmrand2` after removing two random directions. Each adapter is scored on 50 AdvBench prompts drawn with its own seed.

## adaptive.csv

| column | meaning |
|---|---|
| `mode` | `standard`, `spread` (concentration penalty with weight `lam`) or `projected` (top-`k` removal during training) |
| `modules` | `attention` for the four attention projections, `all` for all seven projections |
| `update_norm`, `participation_ratio`, `top2_energy` | means over all adapted modules |
| `detector_score` | participation ratio over rank, averaged over the attention projections |
| `common_*` | the 50 AdvBench prompts drawn with evaluation seed 42, shared by every adapter |
| `own_*` | 50 AdvBench prompts drawn with the adapter's own seed |

`clean`, `attacked` and `repaired` are refusal of the clean model, the attacked model and the attacked model after top-two removal.

## benign_spectra.csv

`source` is the Alpaca slice a benign adapter was trained on. `detector_score` is computed exactly as in `adaptive.csv`.

## mixed_freeze.csv and rank_ladder.csv

`harm_fraction` is the harmful share of 100 training examples, the rest being Alpaca task pairs. `task_nll` is the mean negative log-likelihood of 200 held-out task references and `task_rouge_l` the ROUGE-L of 50 held-out generations. In `rank_ladder.csv`, `adapted` is the fine-tuned model and `rmtopk` the same model after top-k removal.
