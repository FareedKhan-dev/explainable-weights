# results

Every number in the walkthrough comes from a file here. Scripts run from the repository root, with `WSP_PAPER` pointing
at the paper's code. Most write the JSON file of the same name.

## Scripts

| Script | What it does |
|---|---|
| `cells.py` | the article's code cells, in order. results/captures.py runs them in one shared namespace and saves |
| `captures.py` | run the post's driver cells for real, in post order, in one shared namespace (like a notebook), |
| `compare.py` | Qwen3.7-Max's picks beside the paper's own networks, as the article prints them. |
| `compute.py` | how much arithmetic one answer takes, counted by PyTorch's own FLOP counter. |
| `template_step.py` | the paper's template, with its random starting weights, reading the post's running prompt. |
| `template_attention.py` | how the paper's random template shares out attention at the last prompt position. |
| `explain_predicts.py` | an explanation is faithful if, written as a program, it predicts what the network does. |
| `fix_answers.py` | what Qwen3.7-Max's first designs and its fixed design answer for the running prompt. |
| `hero_answers.py` | the output block after the hero, a real capture. |
| `ig_shares.py` | integrated gradients for every character of one sum, from the trained adder (the black box). |
| `blackbox_explain.py` | explain the trained black box (addition) with the usual tools, and record what they say. |
| `blackbox_wall.py` | what the trained black box's weights look like, for the post's first figure. |
| `blackbox_text.py` | real-world black boxes. The paper's template network trained the usual way on TREC and on |
| `train_baseline.py` | train the black-box baseline (addition) and keep its learning curve and weights. |
| `floors.py` | the guessing floor of every set the post scores on, the score of always giving the commonest answer. |
| `data_card.py` | a data card for every set the study scores on, with the leakage checks a reviewer would ask for. |
| `rebuild_scores.py` | the rebuild test's results, every network checked as the originals were. |
| `repeat_picks.py` | score each dataset pick five times, each in a fresh process, and report the spread. |
| `score_snapshot.py` | build one saved network and score it, in its own process. |
| `staircase.py` | the attempt log of Qwen3.7-Max's word reversal run (run 1), as the post prints it. |
| `qwen_network.py` | everything the figures show about the one network Qwen3.7-Max got right. |
| `traces.py` | the worked examples the figures and the post follow, each checked against the real network. |
| `paper_checks.py` | the paper's own networks, checked exactly as Qwen's networks are (results/score_snapshot.py). |
| `audit_repro.py` | the data-use audit, rerun after its post-study split into helpers, must give the same verdicts. |

## Result files

`audit_repro.json`, `audit_transcripts_main2.json`, `baseline_curve.json`, `blackbox_explain.json`, `blackbox_nli-snli.json`, `blackbox_nli-snli_long.json`, `blackbox_paraphrase-mrpc.json`, `blackbox_paraphrase-mrpc_long.json`, `blackbox_sentiment-sst2.json`, `blackbox_sentiment-sst2_long.json`, `blackbox_trec-coarse.json`, `blackbox_trec-coarse_long.json`, `blackbox_wall.json`, `compute.json`, `cost.json`, `data_card.json`, `environment.json`, `explain_predicts.json`, `fix_answers.json`, `floors.json`, `ig_shares.json`, `laundering.json`, `paper_checks.json`, `paper_exhaustive.json`, `qwen_network.json`, `qwen_runs.json`, `rebuild_word-reversal-3x3.json`, `repeat_picks.json`, `template_attention.json`, `template_step.json`, `traces.json`

`qwen_runs.json`, `cost.json`, `environment.json`, `audit_transcripts_main2.json` and `laundering.json` were computed
from the conversation logs, which are not published.

## Printouts and checkpoints

- `outputs/<id>.txt`: what each cell of the walkthrough prints. `captures.py --check` compares a fresh run with them.
- `baseline_addition.step35000.pt`: the trained black box the walkthrough loads (training stopped at step 35,000 of a
  planned 40,000).
- `blackbox_<dataset>.pt` and `blackbox_<dataset>_long.pt`: the black boxes trained on the four text datasets.
