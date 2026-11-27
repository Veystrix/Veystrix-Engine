"""Decode-first admission and cooperative execution on the model thread."""
from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty
import threading
from time import perf_counter

from veystrix.backends.base import GenerationCancelled
from veystrix.request_state import AdmissionController, DecodeFirstQueue, RequestState, StateRefused, DECODE, PREFILL
from veystrix.scheduler import Busy, Scheduler, POLL_SECONDS


@dataclass
class _Quantum:
    state: RequestState
    call: object
    done: threading.Event = field(default_factory=threading.Event)
    value: object = None
    error: BaseException | None = None
    decode_args: tuple | None = None
    cancel: object = None


class BatchScheduler(Scheduler):
    """Continuously admit bounded request states and co-step singleton forwards."""

    def __init__(self, *, max_active=4, headroom=None, **options):
        super().__init__(**options)
        self.max_active = max(1, min(max_active, 4))
        self._headroom = headroom or available_headroom
        self.admission = AdmissionController()
        self.ready = DecodeFirstQueue(clock=self._clock)
        self._quanta = {}
        self._wake = threading.Event()
        self._closing = False
        self._legacy = False
        self._next_memory_check = 0.0
        self._memory_ready = False
        self._decode_groups = 0
        self._max_decode_group = 0
        self._active_slots = threading.BoundedSemaphore(self.max_active)

    def snapshot(self):
        report = super().snapshot()
        report["batch"] = {"max_active": self.max_active, **self.admission.snapshot()}
        with self._lock:
            report["batch"].update(decode_groups=self._decode_groups, max_decode_group=self._max_decode_group)
        return report

    @property
    def busy(self):
        return bool(self.admission.used) or super().busy

    def release(self, state):
        self.admission.release(state.request_id)
        with self._lock:
            self.completed += 1
        self._wake.set()

    def acquire_request(self):
        if not self._active_slots.acquire(blocking=False):
            raise Busy("all continuous batching request slots are occupied")

    def release_request(self):
        self._active_slots.release()

    def reserve(self, request_id, size):
        state = RequestState(request_id, request_id, state_bytes=size, last_active=self._clock())
        with self._lock:
            if self._closing or not self.loop_running or self._legacy:
                raise Busy("the inference executor is unavailable for batching")
            # Use the full outstanding reservation since some of it has
            # not yet materialized in the sampled OS headroom.
            if size + self.admission.used > self._headroom():
                raise Busy("insufficient available memory for another conversation")
            try:
                self.admission.admit(state)
            except StateRefused as error:
                raise Busy(str(error)) from error
        return state

    def submit_quantum(self, state, fn, decode):
        return self._submit(_Quantum(state, fn), decode)

    def submit_decode(self, state, args, cancel):
        return self._submit(_Quantum(state, None, decode_args=args, cancel=cancel), True)

    def _submit(self, work, decode):
        state = work.state
        with self._lock:
            if self._closing:
                raise Busy("the inference scheduler is stopping")
            if state.request_id in self._quanta:
                raise RuntimeError("conversation already has queued model work")
            state.phase = "decode" if decode else "prefill"
            self._quanta[state.request_id] = work
            self.ready.enqueue(state, DECODE if decode else PREFILL)
            self._wake.set()
        work.done.wait()
        if work.error is not None:
            raise work.error
        return work.value

    def _next_quantum(self):
        with self._lock:
            state, _ = self.ready.next_ready()
            return self._quanta.pop(state.request_id) if state else None

    def _decode_group(self, first):
        backend = first.decode_args[0]
        with self._lock:
            states = self.ready.take_where(
                DECODE, lambda state: (
                    (args := self._quanta[state.request_id].decode_args) is not None
                    and backend.batch_group_key() is args[0].batch_group_key()
                ),
                limit=self.max_active - 1,
            )
            return [first, *(self._quanta.pop(state.request_id) for state in states)]

    def _run_decode_group(self, group):
        live = []
        for work in group:
            if work.cancel():
                work.error = GenerationCancelled()
            else:
                live.append(work)
        if not live:
            return
        backend, token, sampler, want_logits = live[0].decode_args
        with backend.batch_scope():
            start = perf_counter()
            if len(live) == 1:
                values = [backend.batch_decode(token, sampler, want_logits)]
            else:
                values = backend.batch_decode_group([work.decode_args for work in live])
            seconds = perf_counter() - start
        for work, value in zip(live, values, strict=True):
            work.value = value, seconds
        with self._lock:
            self._decode_groups += len(live) > 1
            self._max_decode_group = max(self._max_decode_group, len(live))

    def _next_legacy(self):
        with self._lock:
            if self.admission.used:
                return None
            try:
                _, _, ticket = self._jobs.get_nowait()
            except Empty:
                return None
            self._legacy = True
            return ticket

    def serve_jobs(self, stop):
        continuous_active = False
        with self._lock:
            self.loop_running = True
        try:
            while not stop.is_set():
                if continuous_active and not self.admission.used:
                    continuous_active = False
                # Legacy jobs share backend and experiment globals. Execute
                # one only after all cooperative conversations have released.
                ticket = self._next_legacy()
                if ticket is not None:
                    try:
                        self._serve_ticket(ticket)
                    finally:
                        with self._lock:
                            self._legacy = False
                    continue
                work = self._next_quantum()
                if work is None:
                    self._wake.wait(POLL_SECONDS)
                    self._wake.clear()
                    continue
                group = [work] if work.decode_args is None else self._decode_group(work)
                try:
                    now = self._clock()
                    if now >= self._next_memory_check:
                        self._memory_ready = self._headroom() > 0
                        self._next_memory_check = now + 0.25
                    if not self._memory_ready:
                        raise Busy("available memory is below the batching reserve")
                    if not continuous_active:
                        continuous_active = True
                    if work.decode_args is None:
                        work.value = work.call()
                    else:
                        self._run_decode_group(group)
                except KeyboardInterrupt:
                    for entry in group:
                        entry.error = Busy("the inference scheduler is stopping")
                    raise
                except BaseException as error:
                    for entry in group:
                        entry.error = entry.error or error
                finally:
                    for entry in group:
                        entry.done.set()
        finally:
            pass
            with self._lock:
                self._closing = True
                self.loop_running = False
                pending, self._quanta = self._quanta, {}
            for work in pending.values():
                self.ready.drop(work.state.request_id)
                work.error = Busy("the inference scheduler is stopping")
                work.done.set()
            self._fail_pending()


def available_headroom():
    from veystrix.bench.memory import sample

    available = sample()["available_bytes"] or 0
    return max(0, available - 4 * 2**30)
