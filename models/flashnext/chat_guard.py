"""Opt-in chat supervisor using the benchmark's existing probes and safety rules.

On a guard trip the benchmark exits the process: a GPU operation may be in
flight and freeing its buffers from the monitoring thread is not safe.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from models.flashnext import memory_controller as MC
from models.flashnext import memory_policy
from models.flashnext import memory_waiver
from models.flashnext.tests.bench import prefill_memory_harness as H
from models.flashnext.tests.bench.bench_prefill_memory import MemoryTracer, RealProbe


GRACE_FLAG = "FLASHNEXT_GUARD_CANCEL_GRACE_S"
DEFAULT_GRACE_S = 20.0           # longest a swap-growth stop waits for the worker to cancel at a layer boundary
FLUSH_S = 1.5                    # after the worker is idle, time to deliver the error to the client before exiting


def cancel_grace_seconds(environ=None) -> float:
    raw = (os.environ if environ is None else environ).get(GRACE_FLAG)
    if raw in (None, ""):
        return DEFAULT_GRACE_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if 0.0 <= value <= 120.0 else DEFAULT_GRACE_S


_TOOLS = {"vm_stat": "/usr/bin/vm_stat", "sysctl": "/usr/sbin/sysctl", "ps": "/bin/ps", "pmset": "/usr/bin/pmset"}


def _spawn_text(command, *, check=True) -> str:
    """Run a probe command without fork().

    The guard samples about 20 commands a second inside the engine process. fork() of that
    process slowed the C++ Nemotron decode from ~138 to ~173 ms/token. With an absolute
    executable and close_fds=False, subprocess uses posix_spawn, which does not duplicate the
    parent's address space. Python's own descriptors are non-inheritable (PEP 446).
    """
    command = [_TOOLS.get(command[0], command[0]), *command[1:]]
    return subprocess.run(command, capture_output=True, text=True, timeout=5, check=check,
                          close_fds=False).stdout


def _spawn_text_or_empty(command) -> str:
    try:
        return _spawn_text(command, check=False)
    except Exception:
        return ""


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_bool),
    ]


class _VMStats64(ctypes.Structure):
    """Exact layout of mach vm_statistics64_data_t (HOST_VM_INFO64) on macOS."""
    _fields_ = [
        ("free", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("inactive", ctypes.c_uint32),
        ("wire", ctypes.c_uint32),
        ("zero_fill", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable", ctypes.c_uint32),
        ("speculative", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor", ctypes.c_uint32),
        ("throttled", ctypes.c_uint32),
        ("external", ctypes.c_uint32),
        ("internal", ctypes.c_uint32),
        ("uncompressed_in_compressor", ctypes.c_uint64),
    ]


def _host_vm() -> dict | None:
    """One zero-subprocess host_statistics64 read, matching vm_stat's exact semantics."""
    lib = _libc()
    if lib is None:
        return None
    try:
        if not getattr(lib, "_vm_args", False):
            lib.mach_host_self.restype = ctypes.c_uint
            lib.host_statistics64.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_uint)]
            lib._vm_args = True
        st = _VMStats64()
        count = ctypes.c_uint(ctypes.sizeof(st) // 4)
        if lib.host_statistics64(lib.mach_host_self(), 4, ctypes.byref(st), ctypes.byref(count)) != 0:
            return None
        page = os.sysconf("SC_PAGE_SIZE")
        mb = lambda pages: pages * page / 1048576.0  # noqa: E731
        # Mach free_count includes speculative_count; vm_stat reports free = free_count - speculative_count.
        free_pages = max(0, int(st.free) - int(st.speculative))
        return {
            "page_bytes": page,
            "available_mb": round(mb(free_pages + int(st.inactive) + int(st.speculative)), 1),
            "free_mb": round(mb(free_pages), 1),
            "compressor_mb": round(mb(st.compressor), 1),
            "wired_mb": round(mb(st.wire), 1),
            "active_mb": round(mb(st.active), 1),
            "file_backed_mb": round(mb(st.external), 1),
            "anonymous_mb": round(mb(st.internal), 1),
            "counters": {
                "swapin": int(st.swapins), "swapout": int(st.swapouts),
                "pagein": int(st.pageins), "pageout": int(st.pageouts),
                "compress": int(st.compressions), "decompress": int(st.decompressions),
            },
        }
    except Exception:
        return None


def _rss_native() -> int | None:
    lib = _libc()
    if lib is None:
        return None
    try:
        if not getattr(lib, "_pid_args", False):
            lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                         ctypes.c_void_p, ctypes.c_int]
            lib.proc_pidinfo.restype = ctypes.c_int
            lib._pid_args = True
        buf = ctypes.create_string_buffer(256)
        if lib.proc_pidinfo(os.getpid(), 4, 0, buf, len(buf)) <= 0:
            return None
        return int(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint64))[1])
    except Exception:
        return None


_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        try:
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            lib.sysctlbyname.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.sysctlbyname.restype = ctypes.c_int
            _LIBC = lib
        except Exception:
            _LIBC = False
    return _LIBC or None


class StrictProbe(RealProbe):
    def rss_bytes(self):
        rss = _rss_native()
        if rss is not None:
            return rss
        try:
            return int(_spawn_text(["ps", "-o", "rss=", "-p", str(os.getpid())], check=False).strip()) * 1024
        except Exception:
            return None

    def swap_used_mb(self):
        lib = _libc()
        if lib is not None:
            xsu = _XswUsage()
            size = ctypes.c_size_t(ctypes.sizeof(xsu))
            if lib.sysctlbyname(b"vm.swapusage", ctypes.byref(xsu), ctypes.byref(size), None, 0) == 0:
                return xsu.xsu_used / (1024.0 * 1024.0)
        return H.parse_swap_used_mb(_spawn_text_or_empty(["sysctl", "vm.swapusage"]))

    def pressure_level(self):
        lib = _libc()
        if lib is not None:
            level = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(level))
            if lib.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(level), ctypes.byref(size), None, 0) == 0:
                return int(level.value)
        return H.parse_pressure_level(_spawn_text_or_empty(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))

    def machine(self):
        native = _host_vm()
        if native is not None:
            return native
        text = _spawn_text(["vm_stat"])
        required = ("page size of", "Pages free", "Pages inactive", "Pages speculative",
                    "Pages occupied by compressor", "Swapins", "Swapouts")
        if (not re.search(r"page size of \d+ bytes", text) or
                any(part not in text for part in required) or
                any(not re.search(rf"(?m)^\s*{re.escape(label)}\s*:\s*\d+\.", text)
                    for label in required[1:])):
            raise ValueError("incomplete vm_stat output")
        return H.parse_vm_stat(text)


class GuardTracer(MemoryTracer):
    def _run(self):
        try:
            super()._run()
        except BaseException as error:
            reason = f"monitor failed: {type(error).__name__}: {error}"
            try:
                self._write({"guard": reason, "phase": None})
            finally:
                self.on_guard(reason)


class ChatGuard:
    def __init__(self, path, *, probe=None, sampler=None, exit_fn=os._exit, controller=None, clock=time.monotonic):
        self.probe = probe or StrictProbe()
        self.sampler = sampler or H.Sampler(runner=self._read_command)
        self.exit_fn = exit_fn
        self.path = Path(path)
        self.start_sample = None
        self.start_vm = None
        self.watchdog = None
        self.last_power_check = 0.0
        self.tracer = None
        self.closed = False
        self.controller = controller
        self.clock = clock
        self.stop_started = None          # monotonic time of the first stop reason (None: not stopping)
        self.stop_mode = None             # "hard" | "graceful"
        self.stop_reason = None
        self.sleep = time.sleep
        self._last_swap_mb = None
        self._missed_samples: dict[str, int] = {}
        self._stop_lock = threading.Lock()

    @staticmethod
    def _read_command(command):
        return _spawn_text(command)

    def _recheck_after_cache_release(self, row, waived):
        try:
            import mlx.core as mx

            mx.clear_cache()
            active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            cache = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
            fresh = dict(row, mlx_active=int(active()), mlx_cache=int(cache()))
        except Exception:
            fresh = row            # cannot release or read: keep the original stop
        return H.guard_reason(fresh, min_available_gb=0.0 if waived else H.MIN_AVAILABLE_GB,
                              compressor_stop_available_mib=0.0 if waived else H.COMPRESSOR_STOP_AVAILABLE_GIB * 1024)

    def _reason(self, row):
        absolute = memory_policy.is_absolute_swap_policy()
        waived = memory_policy.is_swap_growth_policy() or memory_waiver.active()
        if not absolute:
            for key in ("mlx_active", "mlx_cache", "available_mb", "compressor_mb", "swapout", "swapin"):
                if row.get(key) is None:
                    self._missed_samples[key] = self._missed_samples.get(key, 0) + 1
                    if self._missed_samples[key] >= 3:
                        return f"critical probe unavailable: {key}"
                else:
                    self._missed_samples[key] = 0
            if not row.get("metal_limit_bytes"):
                return "critical probe unavailable: Metal working-set limit"
            reason = H.guard_reason(row, min_available_gb=0.0 if waived else H.MIN_AVAILABLE_GB,
                                    compressor_stop_available_mib=0.0 if waived else H.COMPRESSOR_STOP_AVAILABLE_GIB * 1024)
            if reason and reason.startswith("MLX holds"):
                reason = self._recheck_after_cache_release(row, waived)
            if reason:
                return reason
        now = time.monotonic()
        if self.start_sample is None or now - self.last_power_check >= 1:
            self.last_power_check = now
            sample = self.sampler.sample()
            required = ["power_source", "low_power", "swap_used_mb"]
            if not absolute:
                required.append("pressure_level")
            for key in required:
                if sample.get(key) is None:
                    self._missed_samples[key] = self._missed_samples.get(key, 0) + 1
                    if self._missed_samples[key] >= 3:
                        return f"critical probe unavailable: {key}"
                else:
                    self._missed_samples[key] = 0
            if sample.get("battery_percent") is None:
                power = str(sample.get("power_source", "")).lower()
                if "ac" not in power and "battery" in power:
                    self._missed_samples["battery_percent"] = self._missed_samples.get("battery_percent", 0) + 1
                    if self._missed_samples["battery_percent"] >= 3:
                        return "critical probe unavailable: battery_percent"
            else:
                self._missed_samples["battery_percent"] = 0
            # Thermal check: tolerates desktop Macs or platforms without battery thermal warning
            thermal = self.sampler.runner(["pmset", "-g", "therm"])
            therm_lower = (thermal or "").lower()
            if not thermal or ("error" in therm_lower and "no thermal" not in therm_lower):
                self._missed_samples["thermal"] = self._missed_samples.get("thermal", 0) + 1
                if self._missed_samples["thermal"] >= 3:
                    return "critical probe unavailable: thermal status"
            else:
                self._missed_samples["thermal"] = 0
                if "thermal warning level" not in therm_lower and "no thermal warning" not in therm_lower:
                    self._missed_samples["thermal_warning_status"] = self._missed_samples.get("thermal_warning_status", 0) + 1
                    if self._missed_samples["thermal_warning_status"] >= 3:
                        return "critical probe unavailable: thermal warning status"
                else:
                    self._missed_samples["thermal_warning_status"] = 0
            if sample.get("thermal_speed_limit") is None and "No CPU power status has been recorded" not in (thermal or ""):
                pass
            if sample.get("thermal_warning"):
                return "thermal warning"
            vm = {"swapout": row.get("swapout"), "swapin": row.get("swapin")}
            if self.start_sample is None:
                if row.get("available_mb") is not None and row["available_mb"] < 9 * 1024 and not waived:
                    return "startup available memory is below 9 GiB"
                self.start_sample, self.start_vm = sample, vm
                # The watchdog keeps battery, Low Power Mode, available-memory floor, compressor-with-low-availability and
                # thermal. Pressure and swap are decided by the staged controller below (owner decision 2026-11-08), so
                # the watchdog's immediate pressure-2 stop, its 64 MB swap-growth stop and its "any swap-out" stop are
                # switched off here and replaced by: level 4 held, level >= 2 persisting, +2.5 GiB swap, active swap rate.
                policy = H.SafetyPolicy(pressure_stop_level=99, swap_growth_mb=float("inf"),
                                        swapin_limit_mib=float("inf"), swap_ceiling_only=absolute,
                                        swap_ceiling_mib=(memory_policy.absolute_swap_ceiling_mib() if absolute
                                                          else H.ABSOLUTE_SWAP_CEILING_MIB))
                policy.allow_low_power = os.environ.get("VEYSTRIX_ALLOW_LOW_POWER") == "1"
                # Owner decision 2026-11-09: VEYSTRIX_MIN_BATTERY_PERCENT=0 removes the battery floor for a run (default stays 25).
                try:
                    policy.min_battery_percent = int(os.environ.get("VEYSTRIX_MIN_BATTERY_PERCENT", policy.min_battery_percent))
                except ValueError:
                    pass
                if waived:
                    policy.min_available_gb = 0.0
                    policy.compressor_stop_available_mib = 0.0
                self.watchdog = H.Watchdog(policy, sample)
                if self.controller is None:
                    emit_fn = getattr(self.tracer, "event", None)
                    self.controller = MC.Controller(MC.Policy.from_environment(swap_only=waived), page_bytes=policy.page_bytes,
                                                    emit=emit_fn)
                MC.install(self.controller)
            sample["available_mb"] = row.get("available_mb")
            sample["compressor_mb"] = row.get("compressor_mb")
            comp_start = getattr(self.tracer, "compressor_start_mb", 0.0)
            stop = self.watchdog.check(sample, {"swapout": 0, "swapin": 0},
                                       {"compressor_mb": comp_start})
            controller_stop = self.controller.observe(
                self.clock(), pressure=sample.get("pressure_level"), swap_used_mb=sample["swap_used_mb"],
                swapout_pages=row.get("swapout"), swapin_pages=row.get("swapin"), available_mb=row.get("available_mb"),
                compressor_mb=row.get("compressor_mb"))
            if self.tracer is not None:
                self.tracer.event("watchdog_sample", memory_state=self.controller.state, **sample)
            return stop or controller_stop
        return None

    def _graceful_ok(self, reason) -> bool:
        """Only the owner-approved swap-growth stop is cancelled cooperatively. A blind guard (missing probe, monitor failure),
        thermal, battery and Metal-limit stops keep the immediate exit: waiting could cost the very resource they protect."""
        c = self.controller
        return bool(c is not None and c.stop_reason == reason and reason.startswith("swap grew")
                    and cancel_grace_seconds() > 0)

    def _hard_exit(self, reason, why):
        print(f"FlashNext safety guard stopped the process ({why}): {reason}\nTrace: {self.path}",
              file=sys.stderr, flush=True)
        try:
            from models.flashnext import flight_recorder
            flight_recorder.record("guard_exit", reason=reason, why=why)
        except Exception:
            pass
        self.exit_fn(H.GUARD_EXIT_CODE)

    def _stop(self, reason):
        """First stop wins. Work admission is already closed (the controller's CRITICAL state). For a swap-growth stop with
        a job in flight the process stays up for a bounded grace period so the job can cancel at its next streamed-layer
        boundary (``GenerationCancelled``) and its client gets an actionable 503 instead of a dead socket; it exits as soon as
        the worker is idle (plus a short flush) or when the grace period ends, whichever is first. A Metal call that does not
        return cannot be interrupted from Python: the grace period bounds that delay and the exit then ends it."""
        with self._stop_lock:
            if self.stop_started is not None:
                return
            self.stop_started, self.stop_reason = self.clock(), reason
            graceful = self._graceful_ok(reason) and MC.worker_busy()
            self.stop_mode = "graceful" if graceful else "hard"
        try:
            from models.flashnext import flight_recorder
            flight_recorder.record("guard_stop", reason=reason, mode=self.stop_mode)
        except Exception:
            pass
        if not graceful:
            self._hard_exit(reason, "immediate")
            return
        print(f"FlashNext safety guard: {reason}. Cancelling the in-flight request (up to {cancel_grace_seconds():g}s), "
              f"then exiting. Do not retry automatically.", file=sys.stderr, flush=True)
        threading.Thread(target=self._grace_then_exit, args=(reason,), name="guard-grace", daemon=True).start()

    def _grace_then_exit(self, reason):
        deadline = self.stop_started + cancel_grace_seconds()
        while self.clock() < deadline and MC.worker_busy():
            self.sleep(0.1)
        if MC.worker_busy():
            self._hard_exit(reason, f"worker did not cancel within {cancel_grace_seconds():g}s (uninterruptible call)")
            return
        self.sleep(FLUSH_S)
        self._hard_exit(reason, "worker cancelled")

    def _adaptive_sleep(self, seconds):
        if seconds >= 1.0:
            pressure = self.probe.pressure_level()
            swap = self.probe.swap_used_mb()
            rising = (self._last_swap_mb is not None and swap is not None and
                      swap - self._last_swap_mb > 1.0)
            if (pressure is not None and pressure >= 2) or rising:
                seconds = 0.25
            if swap is not None:
                self._last_swap_mb = swap
        self.sleep(seconds)

    def start(self):
        metal = self.probe.metal_limit()
        self.tracer = GuardTracer(self.probe, self.path, interval=1.0,
                                  guard=self._reason, on_guard=self._stop, metal_limit=metal,
                                  sleep=self._adaptive_sleep,
                                  strict_readings=True,
                                  compressor_warn_mib=H.MAX_COMPRESSOR_GROWTH_GIB * 1024)
        # Synchronous first sample means missing probes block model loading.
        try:
            self.tracer.step()
        except Exception as error:
            reason = f"monitor failed: {type(error).__name__}: {error}"
            try:
                self.tracer._write({"guard": reason, "phase": None})
            finally:
                self._stop(reason)
            raise RuntimeError(reason) from error
        if self.tracer.guard_reason:
            raise RuntimeError(self.tracer.guard_reason)
        from models.flashnext import trace_phase

        trace_phase.register(self.tracer.phase)
        self.tracer.start()
        atexit.register(self.close)
        return self

    def close(self):
        if self.tracer is not None and not self.closed:
            self.closed = True
            self.tracer.event("shutdown", memory_ledger=self.controller.ledger() if self.controller else None)
            if MC.current() is self.controller:
                MC.install(None)
            self.tracer.stop()

