"""wsp/saliency.py: what the usual after-the-fact tools can say about a trained black box.

For one answer character, two standard attributions over the input characters:
  gradient              how strongly that character's score reacts to each input character: the size of the
                        score's gradient with respect to that character's embedding,
  integrated_gradients  (Sundararajan, Taly and Yan, 2017) the gradient averaged along a straight path from a baseline
                        (the same positions with no characters) to the real input, times the input's distance from
                        it. Its attributions add up to the score's change from the baseline, which is checked.
Both answer "which inputs does this answer depend on". Neither says what the network does with them."""
import torch


def from_embeddings(model, h):
    """The rest of the network after the embedding layer (SimpleTransformer)."""
    for block in model.blocks:
        h = block(h)
    return model.head(model.final_ln(h))


def parts(model, ids):
    """Character and position embeddings of one input, apart, so the characters can be scaled alone."""
    pos = model.pos_emb(torch.arange(ids.shape[1]))[None]
    return model.token_emb(ids).detach(), pos.detach()


def score(model, tok, pos, target):
    """The last position's score for the target character."""
    return from_embeddings(model, tok + pos)[:, -1, target]


def gradient(model, ids, target):
    tok, pos = parts(model, ids)
    tok.requires_grad_(True)
    score(model, tok, pos, target).sum().backward()
    return tok.grad[0].norm(dim=-1)                                  # one number per input character


def integrated_gradients(model, ids, target, steps=64):
    """Attributions per input character, and the score change they should add up to."""
    tok, pos = parts(model, ids)
    alphas = ((torch.arange(steps) + 0.5) / steps)[:, None, None]    # the midpoints of the path
    path = (alphas * tok).requires_grad_(True)                       # from no characters to the real ones
    score(model, path, pos.expand(steps, -1, -1), target).sum().backward()
    attr = (tok[0] * path.grad.mean(dim=0)).sum(dim=-1)
    with torch.no_grad():
        gap = (score(model, tok, pos, target) - score(model, torch.zeros_like(tok), pos, target)).item()
    return attr.detach(), gap
