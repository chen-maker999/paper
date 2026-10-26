"""LLMLingua-2 baseline (Pan et al., 2024): learned token-level prompt compression.

A token classifier (the released LLMLingua-2 checkpoints, XLM-RoBERTa or mBERT
fine-tuned on MeetingBank compressions) gives every word a probability of being
kept. As for the other compressors, the task and the last `recent` turns are
kept verbatim; the older history is compressed by keeping the most probable
words, across all older turns, until the budget is met. Line structure is
preserved (a line keeps its surviving words in order; empty lines are dropped).

Word probabilities depend only on a turn's text, so they are computed once per
turn and reused across decision points and budgets.
"""
from __future__ import annotations

import os

import numpy as np

from .compressors import Compressor, _fit_recent, _pinned_units, _split
from .core import Unit, count_tokens

DEFAULT_MODEL = os.environ.get("SKC_LINGUA_MODEL", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
_LOADED = {}


def _load(model_name, device):
    key = (model_name, device)
    if key not in _LOADED:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        torch.set_num_threads(int(os.environ.get("SKC_LINGUA_THREADS", "1")))
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()
        _LOADED[key] = (tok, model)
    return _LOADED[key]


class LLMLingua2(Compressor):
    name = "llmlingua2"

    def __init__(self, model_name=DEFAULT_MODEL, recent=2, device="cpu", max_len=512, batch=8, name=None):
        self.model_name, self.recent, self.device = model_name, recent, device
        self.max_len, self.batch = max_len, batch
        self._probs = {}   # turn text -> (lines as word lists, flat word probabilities)
        if name:
            self.name = name

    # ------------------------------------------------------------------ scoring
    def _score(self, text):
        if text in self._probs:
            return self._probs[text]
        lines = [ln.split() for ln in text.split("\n")]
        words = [w for ln in lines for w in ln]
        probs = np.zeros(len(words), dtype=np.float32)
        if words:
            import torch
            tok, model = _load(self.model_name, self.device)
            # split the word sequence into windows that fit the encoder
            sub = [len(tok.tokenize(w)) or 1 for w in words]
            windows, start, n = [], 0, 0
            for i, c in enumerate(sub):
                if n + c > self.max_len - 2 and i > start:
                    windows.append((start, i)); start, n = i, 0
                n += c
            windows.append((start, len(words)))
            for b in range(0, len(windows), self.batch):
                chunk = windows[b:b + self.batch]
                enc = tok([words[s:e] for s, e in chunk], is_split_into_words=True, truncation=True,
                          max_length=self.max_len, padding=True, return_tensors="pt")
                with torch.no_grad():
                    p = torch.softmax(model(**{k: v.to(self.device) for k, v in enc.items()}).logits, -1)[..., 1]
                p = p.cpu().numpy()
                for j, (s, e) in enumerate(chunk):
                    acc = np.zeros(e - s); cnt = np.zeros(e - s)
                    for k, wid in enumerate(enc.word_ids(j)):
                        if wid is not None:
                            acc[wid] += p[j, k]; cnt[wid] += 1
                    probs[s:e] = acc / np.maximum(cnt, 1)
        costs = np.array([count_tokens(" " + w) for w in words], dtype=np.int32)
        self._probs[text] = (lines, probs, costs)
        return lines, probs, costs

    # ------------------------------------------------------------------ compress
    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        budget -= cost
        tail = hist[len(hist) - self.recent:] if self.recent > 0 else []
        rec, used = _fit_recent(tail, budget)
        rest = hist[: len(hist) - len(tail)]
        room = budget - used
        if room <= 0 or not rest:
            return units + rec
        scored = [(t, *self._score(t.text)) for t in rest]
        order = sorted(((float(pr), ti, wi) for ti, (_, _, probs, _) in enumerate(scored)
                        for wi, pr in enumerate(probs)), key=lambda x: -x[0])
        # newlines and word joins make the rendered size differ slightly from the sum of
        # word costs: shrink the word budget until the rendered context fits
        target = room
        for _ in range(8):
            out = self._select(scored, order, target)
            over = sum(u.tokens for u in out) - room
            if over <= 0:
                break
            target -= over + 8
        return units + out + rec

    @staticmethod
    def _select(scored, order, room):
        keep = [set() for _ in scored]
        spent = 0
        for pr, ti, wi in order:
            c = int(scored[ti][3][wi])
            if spent + c > room:
                continue
            keep[ti].add(wi); spent += c
        out = []
        for ti, (t, lines, _, _) in enumerate(scored):
            if not keep[ti]:
                continue
            text, k = [], 0
            for ln in lines:
                kept = [w for j, w in enumerate(ln) if k + j in keep[ti]]
                k += len(ln)
                if kept:
                    text.append(" ".join(kept))
            out.append(Unit(t.idx, "lingua", "\n".join(text)))
        return out
