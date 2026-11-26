"""Layer-aware expert eviction for caches with forward-ordered routing."""
from collections import OrderedDict


class LeastStale:
    """Prefer prior-pass entries, then layers already visited in this pass.

    SpecMD §4.2: https://arxiv.org/html/2602.03921v1#S4.SS2
    Only demand marks an entry current. Each runtime stages speculative reads
    separately and promotes them when the actual router requests them.
    """

    def __init__(self):
        self._layers = {}
        self._pass = 0
        self._last_layer = -1

    def advance(self, layer):
        if layer < self._last_layer:
            self._pass += 1
        self._last_layer = layer

    def touch(self, key, *, seeded=False):
        lane = self._layers.setdefault(key[0], OrderedDict())
        lane[key] = self._pass - int(seeded)
        lane.move_to_end(key, last=not seeded)

    def discard(self, key):
        lane = self._layers[key[0]]
        del lane[key]
        if not lane:
            del self._layers[key[0]]

    def victim(self, protected):
        # Within a layer only the oldest unprotected entry can win. Search
        # at most one candidate per layer, even for a pool with thousands of slots.
        best, rank = None, None
        for layer, lane in self._layers.items():
            candidate = next((key for key in lane if key not in protected), None)
            if candidate is None:
                continue
            ahead = layer > self._last_layer
            priority = (lane[candidate] == self._pass, ahead, -layer if ahead else layer)
            if rank is None or priority < rank:
                best, rank = candidate, priority
        return best
