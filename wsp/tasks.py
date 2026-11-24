"""wsp/tasks.py: the five problems, from a toy to a real dataset, as character-level prompts with exact answers.

Each generator draws the same examples as the paper's own task code for the same seed (a test checks that), so a
network scored here and a network scored by the paper see identical prompts."""
import calendar
import random
import string
from dataclasses import dataclass
from math import gcd

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


@dataclass
class Example:
    prompt: str
    answer: str


def word_reversal(rng):
    """Three 3-letter words, reversed: cat_dog_pig= gives pig_dog_cat."""
    # Three random 3-letter words
    words = []
    for _ in range(3):
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(3))
        words.append(word)

    # The answer is the same words in reverse
    prompt = "_".join(words) + "="
    answer = "_".join(reversed(words))
    return Example(prompt, answer)


def addition(rng):
    """Two 5-digit numbers and their 6-digit sum: 12345+67890= gives 080235."""
    a, b = rng.randint(0, 99999), rng.randint(0, 99999)
    return Example(f"{a:05d}+{b:05d}=", f"{a + b:06d}")


def gcd_pair(rng):
    """The greatest common divisor of two numbers from 1 to 999: 084,036= gives 012."""
    a, b = rng.randint(1, 999), rng.randint(1, 999)
    return Example(f"{a:03d},{b:03d}=", f"{gcd(a, b):03d}")


def ordinal(d):
    return f"{d}{'th' if 11 <= d <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def date(rng):
    """A date written one of nine common ways, rewritten as YYYY-MM-DD: march 3rd 2021 gives 2021-03-03."""
    fmt = rng.randrange(9)
    y = rng.randint(2000, 2099) if fmt == 8 else rng.randint(1900, 2099)
    m = rng.randint(1, 12)
    d = rng.randint(1, calendar.monthrange(y, m)[1])
    name, abbr = MONTHS[m - 1], MONTHS[m - 1][:3]
    text = [f"{name} {ordinal(d)} {y}", f"{name} {d}, {y}", f"{d} {name} {y}", f"the {ordinal(d)} of {name}, {y}",
            f"{abbr}. {d}, {y}", f"{d}-{abbr}-{y}", f"{m}/{d}/{y}", f"{y}.{m:02d}.{d:02d}", f"{m:02d}/{d:02d}/{y % 100:02d}"][fmt]
    return Example(text.ljust(31, "_") + "=", f"{y:04d}-{m:02d}-{d:02d}")


def clean(text, allowed, length):
    """Lowercase, swap unknown characters for spaces, cut or pad with '_' to a fixed length."""
    t = "".join(c if c in allowed else " " for c in text.lower())
    return " ".join(t.split())[:length].ljust(length, "_")


def trec_pool(size=2000):
    """The first 2,000 training questions of TREC, cleaned, with their coarse label as one digit."""
    from datasets import load_dataset
    ds = load_dataset("CogComp/trec", split="train", revision="refs/convert/parquet")
    allowed = set("abcdefghijklmnopqrstuvwxyz '")
    return [(clean(t, allowed, 80), str(c)) for t, c in zip(ds["text"][:size], ds["coarse_label"][:size])]


PROBLEMS = {                                      # generator, characters, answer length
    "word-reversal-3x3": (word_reversal, "abcdefghijklmnopqrstuvwxyz_=", 11),
    "addition-five-digits": (addition, "0123456789+=", 6),
    "gcd-three-digits": (gcd_pair, "0123456789,=", 3),
    "date-normalization": (date, "abcdefghijklmnopqrstuvwxyz0123456789 ,./-_=", 10),
    "trec-coarse": (None, "abcdefghijklmnopqrstuvwxyz '_=012345", 1),
}


class Problem:
    """One problem: its characters, how text becomes token ids, and seeded examples."""

    def __init__(self, name):
        self.name = name
        self.make, chars, self.answer_len = PROBLEMS[name]
        self.vocab = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = dict(enumerate(self.vocab))
        self.pool = trec_pool() if name == "trec-coarse" else None

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def examples(self, n, seed):
        rng = random.Random(seed)
        if self.pool:                             # questions drawn from the pool, as the paper samples them
            return [Example(t + "=", a) for t, a in rng.sample(self.pool, n)]
        return [self.make(rng) for _ in range(n)]
