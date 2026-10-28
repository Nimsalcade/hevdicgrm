"""A small decoder-only transformer in NumPy, with optional heads that predict a tag for each token.

PyTorch isn't available where this was written, so the backward pass is written out by hand;
tests/test_tinylm.py checks it against finite differences. The tag heads read the same final
hidden state as the next-token head, so a model trained to predict tags runs on raw text alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Params = dict[str, np.ndarray]

_GELU_C = math.sqrt(2 / math.pi)


@dataclass(frozen=True)
class Config:
    vocab: int
    tag_sizes: tuple[int, ...] = ()  # one extra output head per tag set
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 256
    dropout: float = 0.1


def init(cfg: Config, rng: np.random.Generator, dtype: type = np.float32) -> Params:
    std = 0.02
    residual_std = std / math.sqrt(2 * cfg.layers)
    d, f = cfg.width, cfg.hidden
    p = {
        "tok": rng.normal(0, std, (cfg.vocab, d)),
        "pos": rng.normal(0, std, (cfg.context, d)),
        "lnf_g": np.ones(d),
        "lnf_b": np.zeros(d),
        "out_b": np.zeros(cfg.vocab),
    }
    for layer in range(cfg.layers):
        p |= {
            f"{layer}.ln1_g": np.ones(d),
            f"{layer}.ln1_b": np.zeros(d),
            f"{layer}.qkv_w": rng.normal(0, std, (d, 3 * d)),
            f"{layer}.qkv_b": np.zeros(3 * d),
            f"{layer}.o_w": rng.normal(0, residual_std, (d, d)),
            f"{layer}.o_b": np.zeros(d),
            f"{layer}.ln2_g": np.ones(d),
            f"{layer}.ln2_b": np.zeros(d),
            f"{layer}.fc_w": rng.normal(0, std, (d, f)),
            f"{layer}.fc_b": np.zeros(f),
            f"{layer}.proj_w": rng.normal(0, residual_std, (f, d)),
            f"{layer}.proj_b": np.zeros(d),
        }
    for i, size in enumerate(cfg.tag_sizes):
        p[f"tag{i}_w"] = rng.normal(0, std, (d, size))
        p[f"tag{i}_b"] = np.zeros(size)
    return {k: v.astype(dtype) for k, v in p.items()}


def forward(
    p: Params, cfg: Config, x: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Next-token logits (B, T, vocab), tag logits [(B, T, size)], and a cache for `backward`.

    Dropout is applied only when `rng` is given.
    """
    b, t = x.shape
    d, heads = cfg.width, cfg.heads
    dh = d // heads
    h = p["tok"][x] + p["pos"][:t]
    h, drop_emb = _dropout(h, cfg.dropout, rng)
    mask = np.triu(np.full((t, t), -1e9, dtype=h.dtype), k=1)
    layers = []
    for layer in range(cfg.layers):
        c: dict = {}
        a, c["ln1"] = _layernorm(h, p[f"{layer}.ln1_g"], p[f"{layer}.ln1_b"])
        qkv = a @ p[f"{layer}.qkv_w"] + p[f"{layer}.qkv_b"]
        q, k, v = (qkv[..., i * d : (i + 1) * d].reshape(b, t, heads, dh).transpose(0, 2, 1, 3) for i in range(3))
        att = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + mask)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        o, c["drop1"] = _dropout(y @ p[f"{layer}.o_w"] + p[f"{layer}.o_b"], cfg.dropout, rng)
        h = h + o
        m, c["ln2"] = _layernorm(h, p[f"{layer}.ln2_g"], p[f"{layer}.ln2_b"])
        g, c["gelu"] = _gelu(m @ p[f"{layer}.fc_w"] + p[f"{layer}.fc_b"])
        o2, c["drop2"] = _dropout(g @ p[f"{layer}.proj_w"] + p[f"{layer}.proj_b"], cfg.dropout, rng)
        h = h + o2
        c |= {"a": a, "q": q, "k": k, "v": v, "att": att, "y": y, "m": m, "g": g}
        layers.append(c)
    z, lnf = _layernorm(h, p["lnf_g"], p["lnf_b"])
    logits = z @ p["tok"].T + p["out_b"]
    tag_logits = [z @ p[f"tag{i}_w"] + p[f"tag{i}_b"] for i in range(len(cfg.tag_sizes))]
    return logits, tag_logits, {"x": x, "drop_emb": drop_emb, "layers": layers, "lnf": lnf, "z": z}


def backward(p: Params, cfg: Config, cache: dict, dlogits: np.ndarray, dtags: list[np.ndarray]) -> Params:
    """Gradients of the loss with respect to every parameter, given its gradients at the logits."""
    x, z = cache["x"], cache["z"]
    b, t = x.shape
    d, heads, f = cfg.width, cfg.heads, cfg.hidden
    dh = d // heads
    grads: Params = {}
    flat_z = z.reshape(-1, d)
    flat_dlogits = dlogits.reshape(-1, cfg.vocab)
    grads["tok"] = flat_dlogits.T @ flat_z  # the output head shares the token embedding
    grads["out_b"] = flat_dlogits.sum(0)
    dz = dlogits @ p["tok"]
    for i, dtag in enumerate(dtags):
        flat = dtag.reshape(-1, dtag.shape[-1])
        grads[f"tag{i}_w"] = flat_z.T @ flat
        grads[f"tag{i}_b"] = flat.sum(0)
        dz = dz + dtag @ p[f"tag{i}_w"].T
    dhid, grads["lnf_g"], grads["lnf_b"] = _layernorm_back(dz, cache["lnf"])
    for layer in reversed(range(cfg.layers)):
        c = cache["layers"][layer]
        # h_out = h_mid + dropout(gelu(ln2(h_mid) @ fc_w + fc_b) @ proj_w + proj_b)
        do2 = dhid if c["drop2"] is None else dhid * c["drop2"]
        grads[f"{layer}.proj_w"] = c["g"].reshape(-1, f).T @ do2.reshape(-1, d)
        grads[f"{layer}.proj_b"] = do2.reshape(-1, d).sum(0)
        dfc = _gelu_back(do2 @ p[f"{layer}.proj_w"].T, c["gelu"])
        grads[f"{layer}.fc_w"] = c["m"].reshape(-1, d).T @ dfc.reshape(-1, f)
        grads[f"{layer}.fc_b"] = dfc.reshape(-1, f).sum(0)
        dln2, grads[f"{layer}.ln2_g"], grads[f"{layer}.ln2_b"] = _layernorm_back(dfc @ p[f"{layer}.fc_w"].T, c["ln2"])
        dhid = dhid + dln2
        # h_mid = h_in + dropout(attention(ln1(h_in)) @ o_w + o_b)
        do = dhid if c["drop1"] is None else dhid * c["drop1"]
        grads[f"{layer}.o_w"] = c["y"].reshape(-1, d).T @ do.reshape(-1, d)
        grads[f"{layer}.o_b"] = do.reshape(-1, d).sum(0)
        dy = (do @ p[f"{layer}.o_w"].T).reshape(b, t, heads, dh).transpose(0, 2, 1, 3)
        att, q, k, v = c["att"], c["q"], c["k"], c["v"]
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True)) / math.sqrt(dh)
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([g.transpose(0, 2, 1, 3).reshape(b, t, d) for g in (dq, dk, dv)], axis=-1)
        grads[f"{layer}.qkv_w"] = c["a"].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
        grads[f"{layer}.qkv_b"] = dqkv.reshape(-1, 3 * d).sum(0)
        dln1, grads[f"{layer}.ln1_g"], grads[f"{layer}.ln1_b"] = _layernorm_back(dqkv @ p[f"{layer}.qkv_w"].T, c["ln1"])
        dhid = dhid + dln1
    if cache["drop_emb"] is not None:
        dhid = dhid * cache["drop_emb"]
    np.add.at(grads["tok"], x.ravel(), dhid.reshape(-1, d))
    grads["pos"] = np.zeros_like(p["pos"])
    grads["pos"][:t] = dhid.sum(0)
    return grads


def nll(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-position negative log-likelihood of integer targets, and the gradient of its mean."""
    lse = _logsumexp(logits)
    picked = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    grad = np.exp(logits - lse[..., None])
    np.put_along_axis(grad, targets[..., None], np.take_along_axis(grad, targets[..., None], axis=-1) - 1, axis=-1)
    return lse - picked, grad / targets.size


def set_nll(logits: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-position -log P(the label is one of the allowed ones), and the gradient of its mean.

    `allowed` is a boolean mask shaped like `logits`; one True per position gives ordinary
    cross-entropy. Several give the loss SenseBERT uses for a word whose sense isn't known.
    """
    lse = _logsumexp(logits)
    masked = np.where(allowed, logits, -np.inf)
    lse_allowed = _logsumexp(masked)
    grad = np.exp(logits - lse[..., None]) - np.exp(masked - lse_allowed[..., None])
    return lse - lse_allowed, grad / lse.size


def lexicon_loss(p: Params, allowed: np.ndarray) -> tuple[float, Params]:
    """Loss for predicting each vocabulary word's allowed labels from its token embedding alone.

    This teaches the embeddings what a dictionary says about every word, including words the
    training text never shows. `allowed` is a (vocab, labels) mask; the classifier is `lex_w`, `lex_b`.
    Returns the mean loss and gradients for `tok`, `lex_w` and `lex_b`.
    """
    logits = p["tok"] @ p["lex_w"] + p["lex_b"]
    losses, dlogits = set_nll(logits[None], allowed[None])
    dlogits = dlogits[0]
    return float(losses.mean()), {"tok": dlogits @ p["lex_w"].T, "lex_w": p["tok"].T @ dlogits, "lex_b": dlogits.sum(0)}


def logprobs(p: Params, cfg: Config, x: np.ndarray) -> np.ndarray:
    """log P(x[:, i + 1] | x[:, : i + 1]) for every position, shape (B, T - 1), without dropout."""
    logits, _, _ = forward(p, cfg, x[:, :-1])
    losses, _ = nll(logits, x[:, 1:])
    return -losses


class AdamW:
    def __init__(self, params: Params, betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8, weight_decay: float = 0.01):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.decay = {k for k, v in params.items() if v.ndim == 2}  # matrices, not biases or norm gains
        self.t = 0

    def step(self, params: Params, grads: Params, lr: float) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1**self.t, 1 - b2**self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            if k in self.decay:
                params[k] *= 1 - lr * self.weight_decay
            params[k] -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip(grads: Params, max_norm: float) -> float:
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / norm
    return norm


def _dropout(x: np.ndarray, rate: float, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray | None]:
    if rng is None or rate == 0:
        return x, None
    keep = (rng.random(x.shape, dtype=np.float32) >= rate).astype(x.dtype) / (1 - rate)
    return x * keep, keep


def _layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, tuple]:
    centred = x - x.mean(-1, keepdims=True)
    inv = 1 / np.sqrt((centred * centred).mean(-1, keepdims=True) + eps)
    xhat = centred * inv
    return xhat * g + b, (xhat, inv, g)


def _layernorm_back(dy: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xhat, inv, g = cache
    width = dy.shape[-1]
    dg = (dy * xhat).reshape(-1, width).sum(0)
    db = dy.reshape(-1, width).sum(0)
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def _softmax(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _logsumexp(s: np.ndarray) -> np.ndarray:
    top = s.max(-1, keepdims=True)
    return (top + np.log(np.exp(s - top).sum(-1, keepdims=True)))[..., 0]


def _gelu(x: np.ndarray) -> tuple[np.ndarray, tuple]:
    t = np.tanh(_GELU_C * (x + 0.044715 * (x * x * x)))  # x * x * x: x**3 takes NumPy's slow general power
    return 0.5 * x * (1 + t), (x, t)


def _gelu_back(dy: np.ndarray, cache: tuple) -> np.ndarray:
    x, t = cache
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * _GELU_C * (1 + 3 * 0.044715 * x * x))
