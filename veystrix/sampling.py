"""Choosing the next token.

We decoded with `argmax`, which is temperature 0 and none of the rest. Qwen's
model card recommends thinking mode at `temperature=1.0`, `top_p=0.95`,
`top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and instruct mode at
`temperature=0.7`, `top_p=0.80`, `presence_penalty=1.5`.

Greedy isn't an oversight in the benchmarks. They prove a change left the
trajectory alone by comparing token IDs across arms, and sampling makes that
impossible, so greedy stays for measurement and the recommended sampler serves
the chat.

The card also gives the remedy for the problem that sent us here:

    you can adjust the presence_penalty parameter between 0 and 2 to reduce
    endless repetition. However, using a higher value may occasionally result
    in language mixing and a slight decrease in model performance.

Greedy breaks a tie the same way every time, so a model that lands between two
near-equal continuations can stay there. We saw it on a SketchUp Ruby task: it
asked whether one API existed about forty times in identical words, on two
routing profiles and two effort levels.
"""
from __future__ import annotations

from dataclasses import dataclass

# Qwen's recommended thinking-mode values, from the model card's Best Practices.
THINKING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}
# Instruct mode. Kept for reference; the chat runs thinking mode.
INSTRUCT = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}


@dataclass(frozen=True)
class Sampling:
    """One decode's sampling settings. Temperature 0 means greedy."""

    temperature: float = THINKING["temperature"]
    top_p: float = THINKING["top_p"]
    top_k: int = THINKING["top_k"]
    min_p: float = THINKING["min_p"]
    presence_penalty: float = THINKING["presence_penalty"]

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @classmethod
    def greedy_settings(cls) -> "Sampling":
        return cls(temperature=0.0)

    @classmethod
    def from_preferences(cls, values: dict) -> "Sampling":
        return cls(
            temperature=float(values.get("temperature", THINKING["temperature"])),
            top_p=float(values.get("top_p", THINKING["top_p"])),
            top_k=int(values.get("top_k", THINKING["top_k"])),
            min_p=float(values.get("min_p", THINKING["min_p"])),
            presence_penalty=float(
                values.get("presence_penalty", THINKING["presence_penalty"])
            ),
        )

    def describe(self) -> str:
        if self.greedy:
            return "greedy"
        return (
            f"temp {self.temperature:g}  top-p {self.top_p:g}  "
            f"top-k {self.top_k}  min-p {self.min_p:g}  "
            f"presence {self.presence_penalty:g}"
        )


class Sampler:
    """Turn one row of logits into one token id.

    Keeps the ids produced so far, since `presence_penalty` applies to tokens
    already in the reply. Call `reset()` between generations.
    """

    def __init__(self, settings: Sampling | None = None):
        self.settings = settings or Sampling()
        self._seen: set[int] = set()

    def reset(self) -> None:
        self._seen.clear()

    def observe(self, token: int) -> None:
        if self.settings.presence_penalty:
            self._seen.add(int(token))

    def _note_sampled(self, token):
        """Record a sampled token at the sampling boundary.

        generate_step samples the next token before the backend loop
        observes the yielded one, so backend-only observation lets the
        first repetition of a new token escape a nonzero presence penalty.
        Recording here keeps history sequential. Zero-penalty behavior is
        untouched: no synchronization, no history, same trajectory.
        """
        if self.settings.presence_penalty and not self.settings.greedy:
            self.observe(int(token))
        return token

    def distribution(self, logits):
        """The distribution `__call__` draws from, for one row of logits.

        Returns ``(ids, masked)``: ``masked`` holds the logits that survive presence penalty, temperature, top-k, min-p and
        top-p (the rest are -inf); ``ids`` maps its positions to token ids, or is None when ``masked`` spans the vocabulary.
        """
        import mlx.core as mx

        s = self.settings
        row = logits.reshape(-1)
        if s.presence_penalty and self._seen:
            seen = mx.array(sorted(self._seen), dtype=mx.int32)
            hits = mx.zeros(row.shape[0], dtype=row.dtype)
            hits = hits.at[seen].add(
                mx.full((seen.shape[0],), s.presence_penalty, dtype=row.dtype)
            )
            row = row - hits

        row = row.astype(mx.float32) / s.temperature

        if s.top_k and 0 < s.top_k < row.shape[0]:
            kept = mx.argpartition(row, kth=-s.top_k, axis=-1)[-s.top_k:]
            floor = mx.min(mx.take(row, kept, axis=-1))
            row = mx.where(row < floor, -mx.inf, row)
            # Shrink the remaining sorts and reductions to the survivor set.
            # Masked entries are -inf, so they never enter the nucleus; the
            # kept-first-crossing rule below still guarantees a non-empty set.
            kept = mx.sort(kept, axis=-1)
            return kept, self._mask(mx.take(row, kept, axis=-1))
        return None, self._mask(row)

    def _mask(self, small):
        """Apply min-p and top-p to logits and return them with the excluded entries at -inf."""
        import mlx.core as mx

        s = self.settings
        probabilities = mx.softmax(small, axis=-1)

        if s.min_p > 0.0:
            floor = s.min_p * mx.max(probabilities)
            small = mx.where(probabilities < floor, -mx.inf, small)
            probabilities = mx.softmax(small, axis=-1)

        if 0.0 < s.top_p < 1.0:
            order = mx.argsort(-probabilities, axis=-1)
            ordered = mx.take(probabilities, order, axis=-1)
            carried = mx.cumsum(ordered, axis=-1)
            # Keep the first token whose cumulative mass crosses top_p, so the
            # nucleus is never empty when one token already exceeds it.
            keep = carried - ordered < s.top_p
            allowed = mx.zeros(small.shape[0], dtype=mx.bool_)
            allowed = allowed.at[order].add(keep)
            small = mx.where(allowed, small, -mx.inf)
        return small

    def __call__(self, logits):
        """`logits` is the final row, shape (1, vocab) or (vocab,)."""
        import mlx.core as mx

        row = logits.reshape(-1)
        if self.settings.greedy:
            # Shape (1,), not a scalar: the decode loop feeds `token[None]`
            # to the model and needs (1, 1).
            return mx.argmax(row, axis=-1).reshape(1).astype(mx.uint32)

        kept, masked = self.distribution(row)
        drawn = mx.random.categorical(masked)
        token = drawn if kept is None else mx.take(kept, drawn, axis=-1)
        return self._note_sampled(token.reshape(1).astype(mx.uint32))

    def verify_point_mass(self, rows, drafts):
        """Lossless speculative sampling for drafts that come from the context.

        ``rows[i]`` are the logits after the i-th verified input token (``len(rows) == len(drafts) + 1``); the caller has
        already observed the token that preceded row 0. A draft is a point mass, so it is accepted with the probability the
        target gives it, and a rejected position is redrawn from the target with that token removed. The tokens that come
        out follow the same distribution as sampling one at a time. Returns ``(accepted, next_token)``: how many drafts were
        kept, and the token that follows them (a redraw after a rejection, an ordinary draw after the last one).
        """
        import mlx.core as mx

        for index, draft in enumerate(drafts):
            kept, masked = self.distribution(rows[index])
            probabilities = mx.softmax(masked, axis=-1)
            hit = (kept == draft) if kept is not None else (mx.arange(masked.shape[0]) == draft)
            chance = mx.sum(mx.where(hit, probabilities, 0.0))
            draw = mx.random.uniform()
            mx.eval(chance, draw)
            # A draft that already holds (almost) all the mass leaves nothing to redraw from, so it stands.
            if float(draw.item()) < float(chance.item()) or float(chance.item()) > 0.999999:
                self.observe(draft)
                continue
            residual = mx.where(hit, -mx.inf, masked)
            drawn = mx.random.categorical(residual)
            token = drawn if kept is None else mx.take(kept, drawn, axis=-1)
            return index, int(token.item())
        kept, masked = self.distribution(rows[len(drafts)])
        drawn = mx.random.categorical(masked)
        token = drawn if kept is None else mx.take(kept, drawn, axis=-1)
        return len(drafts), int(token.item())
