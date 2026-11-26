"""Bounded, weight-free proposals from previously seen token sequences.

``VEYSTRIX_LOOKUP_V2=1`` switches both runtimes to the drafter described in HyperQwen's "drafting from the context"
(github.com/syv-ai/HyperQwen, docs/optimizations.md): suffix keys up to 12 tokens, the whole prompt as history instead of the
last 2,048 tokens, a longer block only while a copy is running, and a match that may overlap its own suffix so a repeating
pattern is proposed from its own period. Without the flag the drafter is exactly the one that shipped before.
"""
import os

V2_FLAG = "VEYSTRIX_LOOKUP_V2"
V2_HISTORY = 32768
V2_WIDTHS = (2, 3, 4, 6, 8, 12)
STRONG_WIDTH = 8      # a match this long is a copy, not a coincidence


def v2_enabled():
    return os.environ.get(V2_FLAG) == "1"


class PromptLookup:
    def __init__(self, tokens=(), *, history=2048, widths=(2, 3, 4), wrap=False):
        if history < 16:
            raise ValueError("prompt lookup history must hold at least 16 tokens")
        self.history = history
        self.widths = tuple(sorted(widths))
        self.wrap = wrap
        self.tokens = []
        self.index = {}
        self.last_width = 0
        for token in tokens[-history:]:
            self.append(token)

    def append(self, token):
        position = len(self.tokens)
        for width in self.widths:
            if width > position:
                break
            self.index[tuple(self.tokens[position - width:position])] = position
        self.tokens.append(int(token))
        if len(self.tokens) >= 2 * self.history:
            tail = self.tokens[-self.history:]
            self.tokens.clear()
            self.index.clear()
            for previous in tail:
                self.append(previous)

    def propose(self, limit):
        self.last_width = 0
        if limit <= 0:
            return []
        end = len(self.tokens)
        for width in reversed(self.widths):
            if width > end:
                continue
            start = self.index.get(tuple(self.tokens[-width:]))
            if start is None:
                continue
            self.last_width = width
            draft = self.tokens[start:start + limit]
            if self.wrap and len(draft) < limit:
                # The match overlaps the suffix it matched: continue with the period it implies.
                period = end - start
                while len(draft) < limit and period > 0:
                    draft.append(draft[len(draft) - period] if len(draft) >= period else self.tokens[start + len(draft)])
            return draft
        return []

    @property
    def strong(self):
        return self.last_width >= STRONG_WIDTH


def make_lookup(tokens, *, v2=None):
    """The drafter for this request: the long-context one under ``VEYSTRIX_LOOKUP_V2=1``, the old one otherwise."""
    if v2 if v2 is not None else v2_enabled():
        return PromptLookup(tokens, history=V2_HISTORY, widths=V2_WIDTHS, wrap=True)
    return PromptLookup(tokens)


def make_gate(max_depth, *, v2=None):
    """A throughput gate; the long-context mode lets the block grow to ``max_depth`` (else 4)."""
    if v2 if v2 is not None else v2_enabled():
        return ThroughputGate(max_depth=max_depth, double=True)
    return ThroughputGate()


class ThroughputGate:
    """Compare completed tokens per second, with periodic plain samples."""

    def __init__(self, *, max_depth=4, double=False):
        self.plain = self.speculative = None
        self.cooldown = self.trials = self.rounds = 0
        self.acceptance = None
        self.depth = 2
        self.max_depth = max_depth
        self.double = double      # a fully accepted block doubles the next one (2, 4, 8): a copy ramps up fast
        self.rounds_total = self.proposed_total = self.accepted_total = 0

    def allow(self):
        self.rounds += 1
        if self.cooldown:
            self.cooldown -= 1
            return False
        return self.plain is not None and self.rounds % 16 != 0

    def observe(self, speculative, count, seconds, *, proposed=0):
        rate = count / max(seconds, 1e-9)
        if not speculative:
            self.plain = rate if self.plain is None else 0.75 * self.plain + 0.25 * rate
            return
        self.speculative = rate if self.speculative is None else 0.75 * self.speculative + 0.25 * rate
        self.trials += 1
        if proposed:
            fraction = (count - 1) / proposed
            self.rounds_total += 1
            self.proposed_total += proposed
            self.accepted_total += count - 1
            self.acceptance = fraction if self.acceptance is None else 0.75 * self.acceptance + 0.25 * fraction
            if self.acceptance >= 0.80 and count == proposed + 1 and proposed == self.depth:
                self.depth = min(self.max_depth, self.depth * 2 if self.double else self.depth + 1)
            elif self.acceptance < 0.60:
                self.depth = max(1, self.depth - 1)
        too_slow = self.plain is not None and self.speculative < self.plain
        low_acceptance = self.acceptance is not None and self.acceptance < 0.45
        if self.trials >= 2 and (too_slow or low_acceptance):
            self.cooldown = 16
            self.trials = 0
            self.speculative = None
            self.acceptance = None
            self.depth = 2

    def stats(self):
        """Counters for telemetry: verified rounds, drafts proposed and accepted, current block length."""
        return {"rounds": self.rounds_total, "proposed": self.proposed_total, "accepted": self.accepted_total,
                "depth": self.depth, "tokens_per_round": ((self.accepted_total + self.rounds_total) / self.rounds_total
                                                          if self.rounds_total else None)}
