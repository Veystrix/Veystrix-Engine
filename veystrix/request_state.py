"""Independent request state and decode-first scheduling (blueprint CON-01/CON-02).

The server's cooperative executor uses this model-free bookkeeping to
reserve room for live conversations and order decode steps ahead of prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import threading
import time

GIB = 2**30
DEFAULT_MAX_STATE_BYTES = 4 * GIB
DECODE = 0
PREFILL = 1


class StateRefused(RuntimeError):
    """Retriable admission failure; no state was reserved."""


@dataclass
class RequestState:
    """One live conversation's scheduling and memory identity."""

    request_id: str
    session_id: str
    model_id: str = ""
    tokens: list = field(default_factory=list)
    kv_bytes_per_token: int = 0
    state_bytes: int = 0
    phase: str = "queued"
    created: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    failures: int = 0

    def reserve(self, tokens: int) -> int:
        """Record additional KV tokens and return the new total reservation."""
        tokens = max(0, int(tokens))
        self.tokens.append(tokens)
        self.state_bytes += tokens * max(0, int(self.kv_bytes_per_token))
        self.last_active = time.monotonic()
        return self.state_bytes

    def release_tokens(self, tokens: int) -> int:
        tokens = max(0, min(int(tokens), sum(self.tokens)))
        self.state_bytes = max(0, self.state_bytes - tokens * max(0, int(self.kv_bytes_per_token)))
        return self.state_bytes

    def cancel(self) -> None:
        self.phase = "cancelled"
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def snapshot(self) -> dict:
        return {"request_id": self.request_id, "session_id": self.session_id, "model_id": self.model_id,
                "phase": self.phase, "tokens": sum(self.tokens), "state_bytes": self.state_bytes,
                "idle_seconds": round(time.monotonic() - self.last_active, 1),
                "cancelled": self.cancelled}


class AdmissionController:
    """CON-02: a hard cap on the sum of live request state, evaluated before any allocation."""

    def __init__(self, max_state_bytes: int = DEFAULT_MAX_STATE_BYTES, *, clock=time.monotonic):
        if int(max_state_bytes) < 0:
            raise ValueError("max state bytes must be non-negative")
        self.max_state_bytes = int(max_state_bytes)
        self._clock = clock
        self._states: dict[str, RequestState] = {}
        self._lock = threading.Lock()
        self.refused = 0

    @property
    def used(self) -> int:
        with self._lock:
            return sum(state.state_bytes for state in self._states.values())

    def admit(self, state: RequestState) -> RequestState:
        """Reserve ``state``. Raises ``StateRefused`` when the cap would be exceeded."""
        with self._lock:
            used = sum(entry.state_bytes for entry in self._states.values())
            if state.request_id in self._states:
                raise StateRefused(f"request {state.request_id} already admitted")
            if used + state.state_bytes > self.max_state_bytes:
                self.refused += 1
                raise StateRefused(
                    f"live request state would reach {(used + state.state_bytes) / GIB:.2f} GiB; "
                    f"the cap is {self.max_state_bytes / GIB:.2f} GiB")
            self._states[state.request_id] = state
            return state

    def grow(self, request_id: str, tokens: int) -> int:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                raise StateRefused(f"request {request_id} is not admitted")
            extra = max(0, int(tokens)) * max(0, int(state.kv_bytes_per_token))
            used = sum(entry.state_bytes for entry in self._states.values())
            if used + extra > self.max_state_bytes:
                self.refused += 1
                raise StateRefused(
                    f"request {request_id} needs {extra / GIB:.2f} GiB more; live state would exceed "
                    f"the {self.max_state_bytes / GIB:.2f} GiB cap")
            state.reserve(tokens)
            return state.state_bytes

    def release(self, request_id: str) -> RequestState | None:
        with self._lock:
            return self._states.pop(request_id, None)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            state = self._states.get(request_id)
        if state is None:
            return False
        state.cancel()
        return True

    def snapshot(self) -> dict:
        with self._lock:
            states = [state.snapshot() for state in self._states.values()]
            used = sum(state.state_bytes for state in self._states.values())
        return {"max_state_bytes": self.max_state_bytes, "used_bytes": used, "requests": states,
                "refused": self.refused}


class DecodeFirstQueue:
    """CON-01: ready decode steps run before queued prefill chunks.

    Priority is computed when the item is popped, not when it is queued, so a
    prefill that has waited long enough is lifted ahead of a steady decode
    stream exactly once (``aging_seconds * 4``).
    """

    def __init__(self, *, aging_seconds: float = 1.0, clock=time.monotonic):
        self.aging_seconds = float(aging_seconds)
        self._clock = clock
        self._entries: dict[str, tuple] = {}
        self._sequence = itertools.count()
        self._lock = threading.Lock()

    def _priority(self, state: RequestState, kind: int, sequence: int) -> tuple:
        lift = 1 if (kind == PREFILL and self.aging_seconds > 0
                     and self._clock() - state.last_active >= 4 * self.aging_seconds) else 0
        return (kind - lift, sequence)

    def enqueue(self, state: RequestState, kind: int = DECODE) -> None:
        with self._lock:
            self._entries[state.request_id] = (state, kind, next(self._sequence))

    def next_ready(self):
        """Pop the highest-priority runnable (state, kind), skipping cancelled requests."""
        with self._lock:
            while self._entries:
                request_id, (state, kind, sequence) = min(
                    self._entries.items(), key=lambda item: self._priority(item[1][0], item[1][1], item[1][2]))
                del self._entries[request_id]
                if state.cancelled:
                    continue
                state.last_active = self._clock()
                return state, kind
            return None, None

    def drop(self, request_id: str) -> bool:
        with self._lock:
            return self._entries.pop(request_id, None) is not None

    def take_where(self, kind, predicate, *, limit):
        """Drain compatible ready entries in enqueue order for one model step."""
        with self._lock:
            matching = [state for state, phase, _ in self._entries.values()
                        if phase == kind and not state.cancelled and predicate(state)][:limit]
            for state in matching:
                del self._entries[state.request_id]
                state.last_active = self._clock()
            return matching

    def pending(self) -> int:
        with self._lock:
            return len(self._entries)


class RequestRegistry:
    """Session-keyed state ownership with a bounded number of live states (CON-04 prerequisite)."""

    def __init__(self, *, max_states: int = 8):
        self.max_states = max(1, int(max_states))
        self._states: dict[str, RequestState] = {}
        self._lock = threading.Lock()
        self.evicted = 0

    def get(self, session_id: str) -> RequestState | None:
        with self._lock:
            return self._states.get(session_id)

    def create(self, session_id: str, request_id: str, **fields) -> RequestState:
        with self._lock:
            existing = self._states.get(session_id)
            if existing is not None and not existing.cancelled:
                return existing
            if len(self._states) >= self.max_states and session_id not in self._states:
                victim = min((state for state in self._states.values() if state.cancelled),
                             key=lambda state: state.last_active, default=None)
                if victim is None:
                    victim = min(self._states.values(), key=lambda state: state.last_active)
                self._states.pop(victim.session_id, None)
                self.evicted += 1
            state = RequestState(request_id=request_id, session_id=session_id, **fields)
            self._states[session_id] = state
            return state

    def remove(self, session_id: str) -> RequestState | None:
        with self._lock:
            return self._states.pop(session_id, None)

    def idle(self, ttl_seconds: float, *, clock=time.monotonic):
        now = clock()
        with self._lock:
            return [state for state in self._states.values() if now - state.last_active >= ttl_seconds]

    def snapshot(self) -> dict:
        with self._lock:
            return {"max_states": self.max_states, "states": len(self._states), "evicted": self.evicted,
                    "sessions": [state.snapshot() for state in self._states.values()]}
