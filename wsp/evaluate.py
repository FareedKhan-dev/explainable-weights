"""wsp/evaluate.py: the fixed judge. Generate each answer one token at a time, greedily, and count exact matches."""
import torch


@torch.no_grad()
def generate(model, prompt_ids, n_tokens):
    """Greedy decoding: append the highest-scoring character, then feed the longer sequence back in."""
    ids = prompt_ids
    for _ in range(n_tokens):
        next_id = model(ids)[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, prompt_ids.shape[1]:]


def accuracy(model, problem, examples, batch=64):
    """The share of examples whose whole answer string comes out exactly right, and the misses."""
    model.eval()
    hits, misses = 0, []
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        ids = torch.tensor([problem.encode(e.prompt) for e in chunk], dtype=torch.long)
        for e, row in zip(chunk, generate(model, ids, problem.answer_len)):
            guess = problem.decode(row.tolist())
            hits += guess == e.answer
            if guess != e.answer:
                misses.append((e.prompt, e.answer, guess))
    return hits / max(1, len(examples)), misses
