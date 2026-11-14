"""Choose a speculative draft depth that maximises emitted tokens per second.

The controller is pure and deterministic. It keeps decayed success/failure
counts per draft position and exponential moving averages of the measured cost
of each phase of a draft/verify/repair cycle, then picks the depth with the
best modelled rate.
"""
from __future__ import annotations

import math

# Cycles that run at start_depth before the model is trusted; the cycle after
# them runs at depth 0 (when allowed) so the one-row verify cost is measured.
_BOOTSTRAP_CYCLES = 2
# Probe interval scale by how the next deeper depth is modelled against the
# chosen one: (smallest rate ratio, interval multiplier). A probe costs about
# one minus the ratio of a cycle, and tells most when the two are close, so
# near ties are probed twice as often and clear losers a quarter as often.
_PROBE_SCALE = ((0.95, 0.5), (0.85, 1.0), (0.7, 2.0), (0.0, 4.0))


def _finite(value, name, *, minimum=0.0):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _count(value, name, *, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class DepthController:
    """Pick how many tokens to draft per speculative cycle.

    A cycle drafts ``depth`` tokens, verifies ``depth + 1`` rows in one pass,
    and when a draft is rejected replays the accepted prefix plus one token
    through single-token steps. The rate model prices all three phases.

    Acceptance of position i (given positions before it were accepted) is
    estimated from decayed counts and shrunk toward the estimate of position
    i - 1, so a deep position that is rarely tested starts near its neighbour
    and one observation moves it by about 1 / (n + strength). ``alpha`` is the
    weight of the cost averages only.
    """

    def __init__(self, max_depth, *, min_depth=0, start_depth=None, alpha=0.25,
                 prior=0.8, probe_every=8, plain_every=64, decay=0.98,
                 strength=3.0):
        self.max_depth = _count(max_depth, "max_depth", minimum=1)
        self.min_depth = _count(min_depth, "min_depth", minimum=0)
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must not exceed max_depth")
        if start_depth is None:
            start_depth = self.max_depth
        self.start_depth = min(
            self.max_depth,
            max(self.min_depth, _count(start_depth, "start_depth", minimum=0)),
        )
        self.alpha = _finite(alpha, "alpha")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.prior = _finite(prior, "prior")
        if self.prior > 1.0:
            raise ValueError("prior must be in [0, 1]")
        self.decay = _finite(decay, "decay")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.strength = _finite(strength, "strength")
        if self.strength <= 0.0:
            raise ValueError("strength must be positive")
        self.probe_every = _count(probe_every, "probe_every", minimum=1)
        self.plain_every = _count(plain_every, "plain_every", minimum=1)

        self.cycles = 0
        self.successes = [0.0] * self.max_depth
        self.failures = [0.0] * self.max_depth
        self.single = None
        self.row = None
        self.draft = None
        self.repair = None
        # Smallest verify seconds per row seen so far. Stands in for the
        # one-row cost when depth 0 is not allowed and so never runs.
        self._single_floor = None
        self.depth_counts = {}
        self.last_rates = {}
        # Accepted count of the latest cycle at each depth, and the cycle
        # number of the latest probe, for the climb rule.
        self._last_accepted = {}
        self._last_probe = None
        self._last_cycle = None
        # The outcome of the latest probe, for the climb rule.
        self._probing = None
        self._probe_passed = False

    # Model ---------------------------------------------------------------

    def _blend(self, current, sample):
        if current is None:
            return sample
        return current + self.alpha * (sample - current)

    def _single_cost(self):
        return self.single if self.single is not None else self._single_floor

    def acceptance(self):
        """Effective acceptance estimate for each draft position."""
        estimates = []
        previous = self.prior
        for hit, miss in zip(self.successes, self.failures):
            previous = (hit + self.strength * previous) / (
                hit + miss + self.strength
            )
            estimates.append(previous)
        return estimates

    def expected_rate(self, depth):
        """Modelled tokens per second at ``depth``, or None if a cost is unknown."""
        if isinstance(depth, bool) or not isinstance(depth, int) \
                or not 0 <= depth <= self.max_depth:
            raise ValueError("depth must be an integer in [0, max_depth]")
        single = self._single_cost()
        if single is None:
            return None
        if depth and (self.row is None or self.draft is None):
            return None
        repair = self.repair if self.repair is not None else single
        accept = self.acceptance()
        tokens = 1.0
        seconds = single
        if depth:
            seconds += (self.draft + self.row) * depth
        reach = 1.0
        for index in range(depth):
            passed = reach * accept[index]
            tokens += passed
            # Position index + 1 is the first rejection: index drafts are
            # accepted and index + 1 tokens are replayed.
            seconds += reach * (1.0 - accept[index]) * (index + 1) * repair
            reach = passed
        return tokens / max(seconds, 1e-12)

    def observe(self, depth, accepted, *, draft_seconds, verify_seconds,
                repair_seconds=0.0):
        """Fold one completed cycle into the model."""
        depth = _count(depth, "depth", minimum=0)
        if depth > self.max_depth:
            raise ValueError("depth must not exceed max_depth")
        accepted = _count(accepted, "accepted", minimum=0)
        if accepted > depth:
            raise ValueError("accepted must not exceed depth")
        draft_seconds = _finite(draft_seconds, "draft_seconds")
        verify_seconds = _finite(verify_seconds, "verify_seconds")
        repair_seconds = _finite(repair_seconds, "repair_seconds")

        for index in range(accepted):
            self.successes[index] = self.successes[index] * self.decay + 1.0
            self.failures[index] *= self.decay
        if accepted < depth:
            self.successes[accepted] *= self.decay
            self.failures[accepted] = self.failures[accepted] * self.decay + 1.0

        if depth == 0:
            self.single = self._blend(self.single, verify_seconds)
        else:
            floor = verify_seconds / (depth + 1)
            if self._single_floor is None or floor < self._single_floor:
                self._single_floor = floor
        single = self._single_cost()
        if depth:
            self.draft = self._blend(self.draft, draft_seconds / depth)
            if single is not None:
                self.row = self._blend(
                    self.row, max(0.0, (verify_seconds - single) / depth)
                )
            if accepted < depth and repair_seconds > 0.0:
                self.repair = self._blend(
                    self.repair, repair_seconds / (accepted + 1)
                )
        self.depth_counts[depth] = self.depth_counts.get(depth, 0) + 1
        self._last_accepted[depth] = accepted
        self._last_cycle = (depth, accepted)
        if depth == self._probing:
            self._probe_passed = accepted == depth
        self._probing = None

    # Policy --------------------------------------------------------------

    def _best(self):
        single = self._single_cost()
        if single is None:
            self.last_rates = {}
            return self.min_depth
        rates = {
            depth: self.expected_rate(depth)
            for depth in range(self.min_depth, self.max_depth + 1)
        }
        self.last_rates = rates
        if any(rate is None for rate in rates.values()):
            # Row and draft costs come only from cycles that draft; run one
            # at the starting depth (at least one token) to learn them.
            return min(self.max_depth, max(self.start_depth, self.min_depth, 1))
        best = self.min_depth
        for depth in range(self.min_depth + 1, self.max_depth + 1):
            if rates[depth] > rates[best]:
                best = depth
        return best

    def _probe_interval(self, chosen):
        """Cycles between probes of ``chosen + 1``, from the modelled gap."""
        deeper = self.last_rates.get(chosen + 1)
        shallower = self.last_rates.get(chosen)
        if not deeper or not shallower:
            return self.probe_every
        ratio = deeper / shallower
        for floor, scale in _PROBE_SCALE:
            if ratio >= floor:
                return max(1, round(self.probe_every * scale))
        return self.probe_every

    def _climb(self, chosen, number):
        """Whether to probe one deeper than ``chosen`` right now.

        At most every second cycle may be an exploration, and two situations
        qualify. A cycle at ``chosen`` (at least 1) accepted every draft and
        the last cycle at ``chosen + 1`` was not rejected at its final
        position: a run of full acceptances climbs without waiting for the
        probe cadence. Or the probe two cycles ago was fully accepted although
        the estimate for its position is below one half: that success
        contradicts the estimate, so the estimate is tested again at once
        (this is how the controller recovers when acceptance rises).
        """
        if self._last_probe is not None and number - self._last_probe < 2:
            return False
        if (chosen >= 1 and self._last_cycle == (chosen, chosen)
                and self._last_accepted.get(chosen + 1) != chosen):
            return True
        return (self._probe_passed and self._last_probe == number - 2
                and self.acceptance()[chosen] < 0.5)

    def choose(self):
        """Depth for the next cycle."""
        cycle = self.cycles
        self.cycles += 1
        if cycle < _BOOTSTRAP_CYCLES:
            return self.start_depth
        if cycle == _BOOTSTRAP_CYCLES and self.min_depth == 0:
            return 0
        chosen = self._best()
        number = cycle + 1
        interval = self._probe_interval(chosen)
        if self.min_depth == 0 and chosen > 0 and number % self.plain_every == 0:
            return 0
        if chosen < self.max_depth and (
            number % interval == 0 or self._climb(chosen, number)
        ):
            self._last_probe = number
            self._probing = chosen + 1
            return chosen + 1
        return chosen

    def stats(self):
        return {
            "cycles": self.cycles,
            "p": self.acceptance(),
            "successes": [float(value) for value in self.successes],
            "failures": [float(value) for value in self.failures],
            "single": self.single,
            "row": self.row,
            "draft": self.draft,
            "repair": self.repair,
            "depth_counts": {
                int(depth): int(count)
                for depth, count in sorted(self.depth_counts.items())
            },
            "rates": {
                int(depth): rate for depth, rate in self.last_rates.items()
            },
        }
