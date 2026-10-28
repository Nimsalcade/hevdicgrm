"""Train a small GPT on tagged English under one label arm, then score it on BLiMP.

    pip install "torch>=2.1" numpy tokenizers
    python english/train.py --data data/babylm --arm roles+sense-sets --train-tokens 1000000 \
        --seed 0 --blimp blimp/data --out results/english.jsonl

Check the whole pipeline on CPU in about a minute before renting a GPU:

    python english/train.py --data data/babylm --arm roles+sense-sets+lexicon --smoke --blimp blimp/data

The arms mirror the synthetic experiment (src/hevdicgrm/tagexp.py): extra targets are predicted from
the same final hidden state as the next token, so every model is scored on raw text alone.
BLiMP's data is the jsonl files in https://github.com/alexwarstadt/blimp/tree/master/data.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = {
    "plain": set(),
    "roles": {"roles"},
    "sense-sets": {"senses"},
    "roles+sense-sets": {"roles", "senses"},
    "lexicon": {"lexicon"},
    "roles+sense-sets+lexicon": {"roles", "senses", "lexicon"},
    "random-classes": {"random"},  # control: a fixed random class per token, on the tokens that carry senses
}


class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.fc, self.out = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in self.qkv(self.ln1(x)).split(d, dim=-1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, vocab: int, context: int, width: int, layers: int, heads: int, dropout: float, tags: dict[str, int]):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, width), nn.Embedding(context, width)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.lnf, self.drop = nn.LayerNorm(width), nn.Dropout(dropout)
        self.tags = nn.ModuleDict({name: nn.Linear(width, n) for name, n in tags.items()})
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.drop(self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device)))
        for block in self.blocks:
            h = block(h)
        h = self.lnf(h)
        return h @ self.tok.weight.T, h  # the output layer shares the token embedding


def set_nll(logits: torch.Tensor, allowed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions of -log P(the label is one of the allowed ones)."""
    allowed = allowed | ~valid[..., None]  # at invalid positions every label is allowed, so the loss is exactly 0
    losses = torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), -1)
    return (losses * valid).sum() / valid.sum().clamp(min=1)


@torch.no_grad()
def dev_loss(model: GPT, ids: torch.Tensor, context: int, windows: int = 400, batch: int = 64) -> float:
    model.eval()
    starts = torch.arange(0, min(len(ids) - context - 1, windows * context), context, device=ids.device)
    total, count = 0.0, 0
    for i in range(0, len(starts), batch):
        x = ids[starts[i : i + batch, None] + torch.arange(context + 1, device=ids.device)]
        logits, _ = model(x[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
        count += x[:, 1:].numel()
    model.train()
    return total / count


@torch.no_grad()
def sentence_logprobs(model: GPT, tokenizer, eos: int, sentences: list[str], device: str, batch: int = 128) -> np.ndarray:
    """Total log-probability of each sentence, read after an end-of-text token."""
    model.eval()
    totals = []
    for i in range(0, len(sentences), batch):
        encoded = [[eos] + tokenizer.encode(s).ids for s in sentences[i : i + batch]]
        length = max(map(len, encoded))
        x = torch.full((len(encoded), length), eos, dtype=torch.long)
        mask = torch.zeros((len(encoded), length - 1))
        for j, ids in enumerate(encoded):
            x[j, : len(ids)] = torch.tensor(ids)
            mask[j, : len(ids) - 1] = 1
        x, mask = x.to(device), mask.to(device)
        logits, _ = model(x[:, :-1])
        picked = logits.float().log_softmax(-1).gather(-1, x[:, 1:, None])[..., 0]
        totals.append((picked * mask).sum(1).cpu())
    model.train()
    return torch.cat(totals).numpy()


def blimp(model: GPT, tokenizer, eos: int, directory: Path, device: str, limit: int | None) -> dict[str, dict]:
    """Accuracy on each BLiMP paradigm: how often the grammatical sentence gets the higher probability."""
    results = {}
    for path in sorted(directory.glob("*.jsonl")):
        pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
        good = sentence_logprobs(model, tokenizer, eos, [p["sentence_good"] for p in pairs], device)
        bad = sentence_logprobs(model, tokenizer, eos, [p["sentence_bad"] for p in pairs], device)
        results[path.stem] = {"accuracy": float((good > bad).mean()), "phenomenon": pairs[0].get("linguistics_term"), "pairs": len(pairs)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="directory written by prepare.py")
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--train-tokens", type=int, help="use the first N training tokens (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blimp", type=Path, help="directory of BLiMP jsonl files")
    parser.add_argument("--out", type=Path, default=Path("results/english.jsonl"))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=6, help="evaluations without improvement before stopping")
    parser.add_argument("--smoke", action="store_true", help="tiny model, few steps, CPU: checks the pipeline end to end")
    args = parser.parse_args()
    if args.smoke:
        args.width, args.layers, args.heads, args.context = 64, 2, 2, 64
        args.max_steps, args.eval_every, args.train_tokens = 20, 10, min(args.train_tokens or 50_000, 50_000)

    from tokenizers import Tokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    meta = json.loads((args.data / "labels.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    train = np.load(args.data / "train.npz")
    n = min(args.train_tokens or len(train["ids"]), len(train["ids"]))
    ids = torch.from_numpy(train["ids"][:n].astype(np.int64)).to(device)
    roles = torch.from_numpy(train["roles"][:n].astype(np.int64)).to(device)
    sets = torch.from_numpy(train["sets"][:n].astype(np.int64)).to(device)
    dev = torch.from_numpy(np.load(args.data / "dev.npz")["ids"].astype(np.int64)).to(device)

    sense_index = {s: i for i, s in enumerate(meta["senses"])}
    set_table = torch.zeros((max(len(meta["sets"]), 1), len(meta["senses"])), dtype=torch.bool)
    for i, names in enumerate(meta["sets"]):
        set_table[i, [sense_index[s] for s in names]] = True
    set_table = set_table.to(device)
    lex_ids = torch.tensor(sorted(int(k) for k in meta["lexicon"]), dtype=torch.long, device=device)
    lex_table = torch.zeros((len(lex_ids), len(meta["senses"])), dtype=torch.bool)
    for row, token in enumerate(lex_ids.tolist()):
        lex_table[row, [sense_index[s] for s in meta["lexicon"][str(token)] if s in sense_index]] = True
    lex_valid = lex_table.any(1).to(device)  # a lexicon sense never seen in the text has no label slot
    lex_table = lex_table.to(device)
    random_class = torch.randint(0, len(meta["senses"]), (meta["vocab"],), generator=torch.Generator().manual_seed(12345)).to(device)

    arm = ARMS[args.arm]
    tags = {}
    if "roles" in arm:
        tags["roles"] = len(meta["roles"])
    if "senses" in arm:
        tags["senses"] = len(meta["senses"])
    if "lexicon" in arm:
        tags["lexicon"] = len(meta["senses"])
    if "random" in arm:
        tags["random"] = len(meta["senses"])
    model = GPT(meta["vocab"], args.context, args.width, args.layers, args.heads, args.dropout, tags).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    def autocast() -> contextlib.AbstractContextManager:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()

    offsets = torch.arange(args.context + 1, device=device)
    best, best_state, best_step, stale, step = math.inf, None, 0, 0, 0
    curve, start_time = [], time.time()
    for step in range(1, args.max_steps + 1):
        windows = torch.randint(0, n - args.context - 1, (args.batch, 1), device=device) + offsets
        x = ids[windows]
        inputs = windows[:, :-1]
        with autocast():
            logits, h = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            if "roles" in arm:
                loss = loss + F.cross_entropy(model.tags["roles"](h).float().reshape(-1, tags["roles"]), roles[inputs].reshape(-1), ignore_index=-1)
            if "senses" in arm or "random" in arm:
                s = sets[inputs]
                valid = s >= 0
            if "senses" in arm:
                loss = loss + set_nll(model.tags["senses"](h).float(), set_table[s.clamp(min=0)], valid)
            if "random" in arm:
                loss = loss + F.cross_entropy(model.tags["random"](h).float()[valid], random_class[x[:, :-1]][valid])
            if "lexicon" in arm:
                loss = loss + set_nll(model.tags["lexicon"](model.tok.weight[lex_ids]).float(), lex_table, lex_valid)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, step / args.warmup)
        optimizer.step()
        if step % args.eval_every == 0:
            loss_now = dev_loss(model, dev, args.context)
            curve.append((step, round(loss_now, 5)))
            print(f"step {step}: train {loss.item():.4f}, dev {loss_now:.4f}", flush=True)
            if loss_now < best - 1e-4:
                best, best_step, stale = loss_now, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    record = {
        "arm": args.arm,
        "train_tokens": n,
        "seed": args.seed,
        "dev_loss": best,
        "best_step": best_step,
        "steps": step,
        "curve": curve,
        "seconds": round(time.time() - start_time),
        "settings": {k: v for k, v in vars(args).items() if k in {"width", "layers", "heads", "context", "dropout", "batch", "lr", "smoke"}},
    }
    if args.blimp:
        paradigms = blimp(model, tokenizer, meta["eos"], args.blimp, device, limit=20 if args.smoke else None)
        record["blimp"] = paradigms
        record["blimp_mean"] = float(np.mean([p["accuracy"] for p in paradigms.values()])) if paradigms else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"{args.arm}, {n:,} tokens, seed {args.seed}: dev loss {best:.4f} at step {best_step}; BLiMP {record.get('blimp_mean')}")


if __name__ == "__main__":
    main()
