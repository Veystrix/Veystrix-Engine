"""Bounded, weight-free proposals from previously seen token sequences."""


class PromptLookup:
    def __init__(self, tokens=(), *, history=2048):
        if history < 16:
            raise ValueError("prompt lookup history must hold at least 16 tokens")
        self.history = history
        self.tokens = []
        self.index = {}
        for token in tokens[-history:]:
            self.append(token)

    def append(self, token):
        position = len(self.tokens)
        for width in range(2, min(position, 4) + 1):
            self.index[tuple(self.tokens[position - width:position])] = position
        self.tokens.append(int(token))
        if len(self.tokens) >= 2 * self.history:
            tail = self.tokens[-self.history:]
            self.tokens.clear()
            self.index.clear()
            for previous in tail:
                self.append(previous)

    def propose(self, limit):
        if limit <= 0:
            return []
        for width in range(min(4, len(self.tokens)), 1, -1):
            start = self.index.get(tuple(self.tokens[-width:]))
            if start is not None:
                return self.tokens[start:start + limit]
        return []


class ThroughputGate:
    """Compare completed tokens per second, with periodic plain samples."""

    def __init__(self):
        self.plain = self.speculative = None
        self.cooldown = self.trials = self.rounds = 0
        self.acceptance = None
        self.depth = 2

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
            self.acceptance = fraction if self.acceptance is None else 0.75 * self.acceptance + 0.25 * fraction
            if self.acceptance >= 0.80 and count == proposed + 1 and proposed == self.depth:
                self.depth = min(4, self.depth + 1)
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
