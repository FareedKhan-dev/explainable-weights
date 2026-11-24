# runs

What the agent wrote, run by run. Each folder is one run of the loop on one problem with one seed:

- `attempts/NNN_<name>.py`: every network the evaluator scored, in order, exactly as the agent wrote it.
- `attempts/attempts.jsonl`: one line per attempt, with its time, name, logged accuracy, size and SHA-256.
- `results/overall_results.csv`: the evaluator's own log, as the paper's code writes it.

The conversation logs are not published. The walkthrough's network is
`word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py`.

| Run | Networks |
|---|---|
| `addition-five-digits-qwen37-max-s1` | 27 |
| `addition-five-digits-qwen37-max-s2` | 20 |
| `boolean-circuit-5-bits-qwen37-max-s1` | 1 |
| `date-normalization-qwen37-max-s1` | 45 |
| `date-normalization-qwen37-max-s2` | 58 |
| `decimal-to-binary-8bit-qwen37-max-s1` | 3 |
| `edit-distance-8-qwen37-max-s1` | 2 |
| `edit-distance-8-qwen37-max-s2` | 4 |
| `gcd-three-digits-qwen37-max-s1` | 41 |
| `gcd-three-digits-qwen37-max-s2` | 50 |
| `linear-interpolation-two-points-qwen37-max-s1` | 39 |
| `nli-snli-qwen37-max-s1` | 43 |
| `nli-snli-qwen37-max-s2` | 51 |
| `paraphrase-mrpc-qwen37-max-s1` | 58 |
| `paraphrase-mrpc-qwen37-max-s2` | 56 |
| `parity-upto10-bits-qwen37-max-s1` | 1 |
| `sentiment-sst2-qwen37-max-s1` | 45 |
| `sentiment-sst2-qwen37-max-s2` | 46 |
| `smallest-prime-factor-qwen37-max-s1` | 37 |
| `smallest-prime-factor-qwen37-max-s2` | 45 |
| `trec-coarse-qwen37-max-s1` | 50 |
| `trec-coarse-qwen37-max-s2` | 34 |
| `word-reversal-3x3-qwen37-max-s1` | 27 |
| `word-reversal-3x3-qwen37-max-s2` | 45 |

Five runs were stopped early for budget, on problems that were then dropped, and the analysis leaves them out:
`boolean-circuit-5-bits-qwen37-max-s1`, `decimal-to-binary-8bit-qwen37-max-s1`, `edit-distance-8-qwen37-max-s1`, `edit-distance-8-qwen37-max-s2`, `parity-upto10-bits-qwen37-max-s1`.

`rebuild/word-reversal-3x3/` holds the rebuild test: the agent's 151-word description (`describe/description.txt`), the
two rebuilds from it (`r1/`, `r2/`) and the control built from the task alone (`control/`).
