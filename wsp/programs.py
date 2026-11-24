"""wsp/programs.py: explanations written as programs, each read off a network's own weights or its own description.

A program that predicts a network's exact output, also with parts of the network switched off, is a stronger test of
an explanation than an accuracy drop. results/explain_predicts.py runs each one against its network."""

ROUTES = {11: 8, 12: 9, 13: 10, 14: 7, 15: 4, 16: 5, 17: 6, 18: 3, 19: 0, 20: 1, 21: 2}    # Qwen's own routing table
FOLDS = [(64, 4), (16, 4), (4, 4), (1, 4)]                       # the GCD network's folds: (scale, ReLU pieces)


def routing(ids, off=(), steps=11, vocab=28):
    """Qwen's reversal network as a program, read off its weights."""
    ids = list(ids)
    for _ in range(steps):
        p = len(ids) - 1
        score = [0.0] * vocab

        # The residual stream carries the letter at p itself, worth 50
        score[ids[p]] += 50.0

        # The head copies the routed letter, worth 100. With the route off,
        # the 100 spreads evenly over positions 0 to p
        if "values" not in off:
            if p in ROUTES and p not in off:
                score[ids[ROUTES[p]]] += 100.0
            else:
                for j in range(p + 1):
                    score[ids[j]] += 100.0 / (p + 1)

        # The highest score wins, and the lowest id breaks a tie
        ids.append(score.index(max(score)))
    return ids[-steps:]


def euclid(a, b, head_off=None, fold_off=None, rounds=12):
    """The paper's GCD network, from its own description: six parser heads read one digit each, then twelve rounds of
    four triangular folds, the last fold of a round also swapping A and B. head_off drops one digit, fold_off skips
    one fold in every round."""
    parts = [(a // 100, 100), (a // 10 % 10, 10), (a % 10, 1), (b // 100, 100), (b // 10 % 10, 10), (b % 10, 1)]
    A = sum(d * p for i, (d, p) in enumerate(parts[:3]) if i != head_off)
    B = sum(d * p for i, (d, p) in enumerate(parts[3:], 3) if i != head_off)
    for _ in range(rounds):
        for i, (s, k) in enumerate(FOLDS):
            if i == fold_off:
                continue
            fold = A + sum((-1) ** m * 2 * max(0.0, A - m * s * B / 2) for m in range(1, k + 1))
            A, B = (B, fold) if i == len(FOLDS) - 1 else (fold, B)     # the last fold also swaps
    return readout(A + B)


def readout(g):
    """Three digits read from g: X is g/100, then g/10 less ten times the first digit, then g less both. The head's
    parabolas floor X, and a value past 9 reads as 9 (only a broken Euclid gives g above 999)."""
    out, emitted = "", 0
    for scale in (100, 10, 1):
        x = g / scale - emitted * 10
        d = min(9, max(0, int(x + 0.005) if x >= 0 else 0))
        out, emitted = out + str(d), emitted * 10 + d
    return out


def pointer(ids, reads=(9.0, 1.0, -8.0), steps=11, vocab=28, sep=26):
    """The paper's 3-weight reversal network, from its own description. At answer step i (word = i // 4) the query is
    reads[0] + reads[1] * i + reads[2] * word, the earlier position whose 1-based index is nearest it wins (the lowest
    on a tie, as argmax picks), and its token is read with '=' folded onto the separator and written back."""
    ids = list(ids)
    for _ in range(steps):
        p = len(ids) - 1
        i = max(p - 11, 0)
        q = reads[0] + reads[1] * i + reads[2] * (i // 4)
        best = min(range(p + 1), key=lambda pos: ((q - (pos + 1)) ** 2, pos))
        read = min(ids[best], sep)
        ids.append(min(range(vocab), key=lambda v: ((read - min(v, sep)) ** 2, v)))
    return ids[-steps:]
