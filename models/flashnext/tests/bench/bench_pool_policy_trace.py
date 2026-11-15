"""Replay a recorded route trace through expert-pool replacement policies (offline; no model, no SSD).

Answers two questions with measured routing instead of an assumed residency:

* decode: what share of routed experts would a pool of N GB already hold, under the production LRU and
  under alternatives, and how far is that from the offline optimum (Belady, an upper bound no online policy
  reaches)?
* short prefill: what share of a request's prefill experts is resident in the pool left behind by the
  previous request's decode (the premise of ``FLASHNEXT_SHORT_PREFILL_POOL_REUSE``)?

Pool semantics follow ``ExpertPool.assign``: one global pool of slots keyed ``(layer, expert)``; per event
the hits are touched first, then misses take a free slot or a victim, and an entry used by the current event
is never a victim. Prefill events only query the pool (the reuse path never fills it).

Accepts route-trace files with ``events`` rows ``[turn, phase, token, layer, [experts]]`` and ``record_bytes``,
or runtime decode-profile JSON with ``--record-bytes``. ``--collisions`` counts experts resident at the start of
a forward pass that were evicted before their first use in that pass. Results describe the recorded routing.

Every policy also reports modelled demand MB/token: misses * record bytes / decode tokens (a token is a distinct
``(turn, token)`` pair). ``cost-aware`` drives the runtime's ``CostAwarePolicy`` in the runtime's access order.
``--predict`` scores the route-frequency predictor against the routed experts and against last-visit persistence.
``--turns`` restricts the replay to the listed turns (calibration holdout). ``--json`` writes everything computed
to ``results/flashnext/<run>/pool-policy-trace.json``.

``--prefetch-slots N`` models ``FLASHNEXT_EXPERT_PREFETCH``: N staging slots come out of the pool (decode capacity
``slots - N``), and after each decode row up to ``--prefetch-per-layer`` records predicted for the next
``--prefetch-depth`` layers are issued. ASSUMPTION, not a measurement: every issued read completes before its
target layer and never delays a demand read, and ``--prefetch-per-layer`` is how many records the SSD can read in
the idle window between two layers' demand reads. The model is therefore an upper bound on the benefit.

    python -m models.flashnext.tests.bench.bench_pool_policy_trace TRACE.json.gz [--gb 1 2 4]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from models.shared.cost_aware import CostAwarePolicy, CostWeights, RouteFrequency
from veystrix.results import output_path

POLICIES = ("lru", "arc", "bip8", "s3fifo", "least-stale", "cost-aware", "belady")
PREDICT_KS = (4, 8, 12, 16)
LAYERS, EXPERTS = 48, 512


class Lru:
    """Production policy (``ExpertPool.assign``)."""

    def __init__(self, slots, insert_cold_every=0):
        self.slots, self.pool, self.every, self.count = slots, OrderedDict(), insert_cold_every, 0

    def contains(self, key):
        return key in self.pool

    def access(self, layer, experts):
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.pool:
                self.pool.move_to_end(key)
                hits += 1
        for key in keys:
            if key in self.pool:
                continue
            if len(self.pool) >= self.slots:
                victim = next((k for k in self.pool if k not in current), None)
                if victim is None:
                    continue
                del self.pool[victim]
            self.pool[key] = 1
            if self.every:                       # BIP: most inserts enter at the cold end
                self.count += 1
                if self.count % self.every:
                    self.pool.move_to_end(key, last=False)
        return hits


class S3Fifo:
    """Replay the runtime eviction policy without allocating expert tensors."""

    def __init__(self, slots):
        from models.flashnext.expert_pool import _S3FIFO

        self.slots = slots
        self.policy = _S3FIFO(slots)

    def contains(self, key):
        return key in self.policy.small or key in self.policy.main

    def access(self, layer, experts):
        keys = list(dict.fromkeys((layer, expert) for expert in experts))
        if len(keys) > self.slots:  # the runtime streams this batch without mutating the pool
            return 0
        protected = set(keys)
        misses = []
        for key in keys:
            if self.contains(key):
                self.policy.hit(key)
            else:
                misses.append(key)
        for key in misses:
            if len(self.policy.small) + len(self.policy.main) == self.slots:
                self.policy.victim(protected, self.slots)
            self.policy.insert(key)
        return len(keys) - len(misses)


class LeastStaleReplay:
    def __init__(self, slots):
        from models.shared.least_stale import LeastStale

        self.slots, self.resident, self.policy = slots, set(), LeastStale()

    def contains(self, key):
        return key in self.resident

    def access(self, layer, experts):
        keys = list(dict.fromkeys((layer, expert) for expert in experts))
        if len(keys) > self.slots:
            return 0
        self.policy.advance(layer)
        protected = set(keys)
        misses = []
        for key in keys:
            if key in self.resident:
                self.policy.touch(key)
            else:
                misses.append(key)
        for key in misses:
            if len(self.resident) == self.slots:
                victim = self.policy.victim(protected)
                self.policy.discard(victim)
                self.resident.remove(victim)
            self.resident.add(key)
            self.policy.touch(key)
        return len(keys) - len(misses)


class CostAwareReplay:
    """Drive the runtime ``CostAwarePolicy`` in the runtime's order: advance, hits, victims, inserts, observe."""

    def __init__(self, slots, weights=None, layers=LAYERS, experts=EXPERTS):
        self.slots, self.policy = slots, CostAwarePolicy(slots, layers=layers, experts=experts, weights=weights)
        self.frequency = self.policy.frequency
        self.slot_of, self.free = {}, list(range(slots - 1, -1, -1))

    def contains(self, key):
        return key in self.slot_of

    def access(self, layer, experts):
        keys = list(dict.fromkeys((layer, expert) for expert in experts))
        if len(keys) > self.slots:
            return 0
        policy = self.policy
        policy.advance(layer)
        misses = []
        protected = []
        for key in keys:
            slot = self.slot_of.get(key)
            if slot is None:
                misses.append(key)
            else:
                policy.touch(key, slot)
                protected.append(slot)
        need = len(misses) - len(self.free)
        if need > 0:
            for slot in policy.victims(need, protected=protected):
                key = policy.key_of(slot)
                policy.discard(key, slot)
                del self.slot_of[key]
                self.free.append(slot)
        for key in misses:
            slot = self.free.pop()
            self.slot_of[key] = slot
            policy.insert(key, slot)
        policy.observe(layer, [key[1] for key in keys], misses=len(misses))
        return len(keys) - len(misses)


class Arc:
    def __init__(self, slots):
        self.n, self.p = slots, 0
        self.t1, self.t2, self.b1, self.b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def contains(self, key):
        return key in self.t1 or key in self.t2

    def _replace(self, in_b2, current):
        while len(self.t1) + len(self.t2) >= self.n:
            use_t1 = self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p) or not self.t2)
            for seg, ghost in ((self.t1, self.b1), (self.t2, self.b2)) if use_t1 else ((self.t2, self.b2), (self.t1, self.b1)):
                victim = next((k for k in seg if k not in current), None)
                if victim is not None:
                    del seg[victim]
                    ghost[victim] = 1
                    while len(self.b1) > self.n:
                        self.b1.popitem(last=False)
                    while len(self.b2) > self.n:
                        self.b2.popitem(last=False)
                    break
            else:
                return

    def access(self, layer, experts):
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.t1:
                del self.t1[key]
                self.t2[key] = 1
                hits += 1
            elif key in self.t2:
                self.t2.move_to_end(key)
                hits += 1
        for key in keys:
            if self.contains(key):
                continue
            if key in self.b1:
                self.p = min(self.n, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
                del self.b1[key]
                self._replace(False, current)
                self.t2[key] = 1
            elif key in self.b2:
                self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
                del self.b2[key]
                self._replace(True, current)
                self.t2[key] = 1
            else:
                self._replace(False, current)
                self.t1[key] = 1
        return hits


class Belady:
    """Offline optimum for this event order: evict the entry used furthest in the future."""

    def __init__(self, slots, events):
        self.slots, self.pool, self.index = slots, {}, 0
        later, self.next_use = {}, [None] * len(events)
        for i in range(len(events) - 1, -1, -1):
            layer, experts = events[i]
            self.next_use[i] = {(layer, e): later.get((layer, e), 1 << 60) for e in experts}
            for e in experts:
                later[(layer, e)] = i

    def contains(self, key):
        return key in self.pool

    def access(self, layer, experts):
        row = self.next_use[self.index]
        self.index += 1
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.pool:
                self.pool[key] = row[key]
                hits += 1
        for key in keys:
            if key in self.pool:
                continue
            if len(self.pool) >= self.slots:
                victim = max((k for k in self.pool if k not in current), key=self.pool.get, default=None)
                if victim is None:
                    continue
                del self.pool[victim]
            self.pool[key] = row[key]
        return hits


def make_policy(name, slots, decode_events, weights=None):
    if name == "cost-aware":
        layers = max([LAYERS] + [layer + 1 for layer, _ in decode_events])
        experts = max([EXPERTS] + [e + 1 for _, row in decode_events for e in row])
        return CostAwareReplay(slots, weights, layers, experts)
    if name == "lru":
        return Lru(slots)
    if name == "bip8":
        return Lru(slots, insert_cold_every=8)
    if name == "arc":
        return Arc(slots)
    if name == "s3fifo":
        return S3Fifo(slots)
    if name == "least-stale":
        return LeastStaleReplay(slots)
    if name == "belady":
        return Belady(slots, decode_events)
    raise ValueError(name)


def load_trace(path, record_bytes=None):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    if "events" in data:
        return data["events"], int(data["record_bytes"])
    if not record_bytes or record_bytes <= 0:
        raise ValueError("a runtime decode profile needs --record-bytes")
    events = [[0, "decode", token["token"], layer, token["routes"][str(layer)]]
              for token in data["tokens"] for layer in token["layer_ids"]]
    return events, record_bytes


def filter_turns(events, turns):
    """Events of the listed turn numbers (all events when ``turns`` is None)."""
    if turns is None:
        return events
    keep = set(turns)
    return [event for event in events if event[0] in keep]


def decode_token_count(events):
    return len({(turn, token) for turn, phase, token, _, _ in events if phase == "decode"})


@dataclass(frozen=True)
class PrefetchConfig:
    slots: int = 0
    depth: int = 2
    width: int = 2
    min_p: float = 0.5
    per_layer: int = 1
    alpha: float = 0.3          # decay of the model's own route table; a cost-aware pool shares its own instead


class StagingModel:
    """Model of the decode prefetch: staged records cost one read when issued and none when demanded."""

    def __init__(self, pool, config, frequency, observes):
        self.pool, self.config, self.frequency, self.observes = pool, config, frequency, observes
        self.staged = OrderedDict()                  # key -> its target layer has been visited; insertion = issue order
        self.next_layer, self.previous = {}, None
        self.issued = self.used = self.displaced = 0

    def covered(self, keys):
        """Remove and count staged records the coming row demands but the pool does not hold."""
        covered = [key for key in keys if key in self.staged and not self.pool.contains(key)]
        for key in covered:
            del self.staged[key]
        self.used += len(covered)
        return len(covered)

    def after(self, layer, experts):
        if self.observes:
            self.frequency.observe(layer, experts)
        if self.previous is not None:
            self.next_layer[self.previous] = layer
        self.previous = layer
        for key in self.staged:
            if key[0] == layer:
                self.staged[key] = True
        self._issue(layer)

    def _targets(self, layer):
        targets, seen = [], {layer}
        for _ in range(self.config.depth):
            layer = self.next_layer.get(layer)
            if layer is None or layer in seen:
                break
            seen.add(layer)
            targets.append(layer)
        return targets

    def _slot(self):
        if len(self.staged) < self.config.slots:
            return True
        oldest = next((key for key, passed in self.staged.items() if passed), None)
        if oldest is None:
            return False
        del self.staged[oldest]
        self.displaced += 1
        return True

    def _issue(self, layer):
        config, budget = self.config, self.config.per_layer
        for target in self._targets(layer):
            candidates = [(target, e) for e, p in self.frequency.predict(target, 64)
                          if p >= config.min_p and (target, e) not in self.staged
                          and not self.pool.contains((target, e))][:config.width]
            for key in candidates:
                if budget <= 0 or not self._slot():
                    return
                self.staged[key] = False
                self.issued += 1
                budget -= 1

    def summary(self):
        return {"issued": self.issued, "used": self.used,
                "wasted": self.displaced + len(self.staged), "displaced": self.displaced,
                "unused_at_end": len(self.staged)}


def replay_detail(events, slots, policy, *, weights=None, collisions=None, prefetch=None):
    """Replay one policy over ``events``; return the counts that every reported figure derives from.

    With ``prefetch`` (slots > 0) the pool decodes with ``slots - prefetch.slots`` records and stages the rest."""
    staging_slots = prefetch.slots if prefetch else 0
    if staging_slots and not 0 < staging_slots < slots:
        raise ValueError(f"{staging_slots} staging slots need a pool larger than that ({slots} slots)")
    capacity = slots - staging_slots
    decode = [(layer, tuple(experts)) for _, phase, _, layer, experts in events if phase == "decode"]
    pool = make_policy(policy, capacity, decode, weights)
    staging = None
    if staging_slots:
        frequency = getattr(pool, "frequency", None)
        observes = frequency is None
        if observes:
            frequency = RouteFrequency(max([LAYERS] + [layer + 1 for layer, _ in decode]),
                                       max([EXPERTS] + [e + 1 for _, row in decode for e in row]),
                                       prefetch.alpha)
        staging = StagingModel(pool, prefetch, frequency, observes)
    decode_hits = decode_total = prefill_hits = prefill_total = 0
    at_pass_start = set()
    previous_pass = None
    future = {}
    if collisions is not None:
        for turn, phase, token, layer, experts in events:
            if phase == "decode":
                future.setdefault((turn, token), set()).update((layer, expert) for expert in experts)
    for turn, phase, token, layer, experts in events:
        if phase == "decode":
            if collisions is not None:
                forward = turn, token
                if forward != previous_pass:
                    at_pass_start = {key for key in future[forward] if pool.contains(key)}
                    previous_pass = forward
                for expert in dict.fromkeys(experts):
                    key = layer, expert
                    if key in at_pass_start and not pool.contains(key):
                        collisions.append(key)
                    at_pass_start.discard(key)
            if staging is not None:
                keys = list(dict.fromkeys((layer, expert) for expert in experts))
                if len(keys) <= capacity:
                    staging.covered(keys)
            decode_hits += pool.access(layer, experts)
            decode_total += len(experts)
            if staging is not None:
                staging.after(layer, experts)
        else:
            prefill_hits += sum(pool.contains((layer, e)) for e in experts)
            prefill_total += len(experts)
    detail = {"decode_hits": decode_hits, "decode_total": decode_total, "misses": decode_total - decode_hits,
              "prefill_hits": prefill_hits, "prefill_total": prefill_total,
              "decode_hit_rate": decode_hits / decode_total if decode_total else 0.0,
              "prefill_resident_rate": prefill_hits / prefill_total if prefill_total else 0.0,
              "decode_capacity": capacity}
    if staging is not None:
        detail["prefetch"] = staging.summary()
    return detail


def replay(events, slots, policy, *, collisions=None, weights=None):
    """Return (decode_hit_rate, prefill_resident_rate) for one policy. Prefill rows only query."""
    detail = replay_detail(events, slots, policy, weights=weights, collisions=collisions)
    return detail["decode_hit_rate"], detail["prefill_resident_rate"]


def mb_per_token(count, record_bytes, tokens):
    return count * record_bytes / tokens / 1e6 if tokens else 0.0


def byte_metrics(detail, record_bytes, tokens):
    """Modelled MB/token (10^6 bytes). Without prefetch every miss is one blocking demand read."""
    misses = detail["misses"]
    out = {key: detail[key] for key in ("decode_hit_rate", "prefill_resident_rate", "decode_hits", "decode_total",
                                        "misses", "decode_capacity")}
    out["demand_mb_per_token"] = mb_per_token(misses, record_bytes, tokens)
    pre = detail.get("prefetch")
    if pre is not None:
        issued, used = pre["issued"], pre["used"]
        out["prefetch"] = pre | {
            "precision": used / issued if issued else 0.0,
            "covered_share": used / misses if misses else 0.0,
            "blocking_mb_per_token": mb_per_token(misses - used, record_bytes, tokens),
            "physical_mb_per_token": mb_per_token(misses + issued - used, record_bytes, tokens)}
    return out


def predictor_quality(events, alpha=0.3, ks=PREDICT_KS):
    """Score ``RouteFrequency.predict`` and last-visit persistence against each decode row's routed experts."""
    rows = [(layer, set(experts)) for _, phase, _, layer, experts in events if phase == "decode"]
    layers = max([LAYERS] + [layer + 1 for layer, _ in rows])
    experts = max([EXPERTS] + [e + 1 for _, row in rows for e in row])
    table = RouteFrequency(layers, experts, alpha)
    last = {}
    counts = {k: [0, 0] for k in ks}                # predicted, correct
    persistence, routed = [0, 0], 0
    for layer, row in rows:
        routed += len(row)
        for k in ks:
            predicted = {e for e, _ in table.predict(layer, k)}
            counts[k][0] += len(predicted)
            counts[k][1] += len(predicted & row)
        previous = last.get(layer, set())
        persistence[0] += len(previous)
        persistence[1] += len(previous & row)
        last[layer] = row
        table.observe(layer, row)

    def score(predicted, correct):
        return {"predicted": predicted, "correct": correct, "routed": routed,
                "precision": correct / predicted if predicted else 0.0,
                "recall": correct / routed if routed else 0.0}

    return {"alpha": alpha, "rows": len(rows), "routed": routed,
            "frequency": {str(k): score(*counts[k]) for k in ks}, "persistence": score(*persistence)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sizes(events, record, gbs, policies, *, weights=None, collisions=False, prefetch=None):
    """Per pool size and policy: the metrics dict (and a prefetch variant at equal total bytes)."""
    tokens = decode_token_count(events)
    sizes = []
    for gb in gbs:
        slots = int(gb * 1e9) // record
        results = {}
        for name in policies:
            found = [] if collisions else None
            entry = byte_metrics(replay_detail(events, slots, name, weights=weights, collisions=found), record, tokens)
            if found is not None:
                entry["collision_misses"] = len(found)
            if prefetch and prefetch.slots:
                entry["with_prefetch"] = byte_metrics(
                    replay_detail(events, slots, name, weights=weights, prefetch=prefetch), record, tokens)
            results[name] = entry
        sizes.append({"gb": gb, "slots": slots, "policies": results})
    return sizes


def format_row(name, entry):
    text = (f"{name} decode {entry['decode_hit_rate']:.1%} / prefill-resident {entry['prefill_resident_rate']:.1%}"
            f" / demand {entry['demand_mb_per_token']:.1f} MB/token")
    if "collision_misses" in entry:
        text += f" / collision misses {entry['collision_misses']}"
    return text


def format_prefetch(name, entry):
    p = entry["prefetch"]
    return (f"  {name} + prefetch (decode {entry['decode_capacity']} slots): hit {entry['decode_hit_rate']:.1%}, "
            f"issued {p['issued']} used {p['used']} wasted {p['wasted']}, precision {p['precision']:.1%}, "
            f"covered {p['covered_share']:.1%} of misses, blocking {p['blocking_mb_per_token']:.1f} MB/token, "
            f"physical {p['physical_mb_per_token']:.1f} MB/token")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--record-bytes", type=int, help="bytes per expert for a runtime decode profile JSON")
    parser.add_argument("--collisions", action="store_true", help="report misses on experts resident at pass start")
    parser.add_argument("--gb", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--weights", help="cost-aware weights JSON (CostWeights.to_mapping format)")
    parser.add_argument("--turns", type=int, nargs="+", help="replay only events of these turn numbers")
    parser.add_argument("--predict", action="store_true", help="score the route-frequency predictor")
    parser.add_argument("--predict-alpha", type=float, default=0.3)
    parser.add_argument("--prefetch-slots", type=int, default=0,
                        help="model N staging slots carved from the pool (0 disables the prefetch model)")
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--prefetch-width", type=int, default=2)
    parser.add_argument("--prefetch-min-p", type=float, default=0.5)
    parser.add_argument("--prefetch-per-layer", type=int, default=1,
                        help="records issued per decode row; an assumption about SSD idle time, not a measurement")
    parser.add_argument("--json", action="store_true", help="write results/flashnext/<run>/pool-policy-trace.json")
    args = parser.parse_args(argv)
    weights = None
    if args.weights:
        try:
            weights = CostWeights.load(args.weights)
        except ValueError as error:
            parser.error(str(error))
    prefetch = PrefetchConfig(args.prefetch_slots, args.prefetch_depth, args.prefetch_width,
                              args.prefetch_min_p, args.prefetch_per_layer)
    if prefetch.slots < 0 or (prefetch.slots and min(prefetch.depth, prefetch.width, prefetch.per_layer) < 1):
        parser.error("prefetch slots must be >= 0 and depth, width and per-layer >= 1")
    try:
        events, record = load_trace(args.trace, args.record_bytes)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    events = filter_turns(events, args.turns)
    tokens = decode_token_count(events)
    decode_rows = sum(1 for event in events if event[1] == "decode")
    label = f", turns {' '.join(map(str, args.turns))}" if args.turns else ""
    print(f"{len(events)} events{label}, {decode_rows} decode rows, {tokens} decode tokens, "
          f"record {record / 1e6:.2f} MB (routing of one recorded session)")
    if prefetch.slots:
        print(f"prefetch model: {prefetch.slots} staging slots, depth {prefetch.depth}, width {prefetch.width}, "
              f"min p {prefetch.min_p:g}, {prefetch.per_layer} issue(s) per row. Assumes an issued read always "
              "completes before its target layer and never delays demand reads: an upper bound on the benefit.")
    try:
        sizes = run_sizes(events, record, args.gb, args.policies, weights=weights, collisions=args.collisions,
                          prefetch=prefetch)
    except ValueError as error:
        parser.error(str(error))
    for size in sizes:
        print(f"{size['gb']:g} GB = {size['slots']} slots: "
              + "; ".join(format_row(name, entry) for name, entry in size["policies"].items()))
        for name, entry in size["policies"].items():
            if "with_prefetch" in entry:
                print(format_prefetch(name, entry["with_prefetch"]))
    predictor = None
    if args.predict:
        predictor = predictor_quality(events, args.predict_alpha)
        print(f"route predictor (alpha {args.predict_alpha:g}, {predictor['rows']} decode rows, "
              f"{predictor['routed']} routed experts)")
        for k, score in predictor["frequency"].items():
            print(f"  frequency top-{k}: precision {score['precision']:.3f} recall {score['recall']:.3f}")
        persistence = predictor["persistence"]
        print(f"  last-visit persistence: precision {persistence['precision']:.3f} recall {persistence['recall']:.3f}")
    if args.json:
        report = {"trace": str(Path(args.trace).resolve()), "trace_sha256": file_sha256(args.trace),
                  "record_bytes": record, "turns": args.turns, "decode_rows": decode_rows, "decode_tokens": tokens,
                  "weights": (weights or CostWeights()).to_mapping(),
                  "weights_path": args.weights, "policies": args.policies,
                  "prefetch_model": ({**prefetch.__dict__, "assumption":
                                      "issued reads complete before their target layer and never delay demand "
                                      "reads; per_layer is an assumed idle-window capacity; upper bound"}
                                     if prefetch.slots else None),
                  "predictor": predictor, "sizes": sizes}
        target = output_path("flashnext", "pool-policy-trace", "pool-policy-trace.json")
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
