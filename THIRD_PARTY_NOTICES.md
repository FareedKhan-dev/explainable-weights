# Third-party notices

## Interpretable weight-space programming (the paper's code)

Source: https://github.com/csinva/weight-space-programming-autoresearch (commit 4f8eb615).

`wsp/model.py`, `wsp/tasks.py` and `wsp/evaluate.py` are ports of the paper's `src/` code, tested equal to it by
`tests/test_ports.py`. Every network file under `runs/` was written by an agent starting from the paper's template,
`experiments/evolve/interpretable_transformer.py`, and keeps its structure. The paper's code is used under this license:

```
MIT License

Copyright (c) 2026 Chandan Singh and Yash Mehta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

The text datasets are not included in this repository. The scripts download them from the Hugging Face Hub, under
their own terms. See each dataset's card:

- TREC question classification: `CogComp/trec`
- SST-2: `stanfordnlp/sst2`
- MRPC: `nyu-mll/glue`, configuration `mrpc`
- SNLI: `stanfordnlp/snli` (CC BY-SA 4.0)
