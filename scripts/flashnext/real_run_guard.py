"""Run one command under the owner's real-run safety rules and stop it when one is broken.

    guard.py --log trace.jsonl [--floor-gib 3] [--swap-growth-mib 512] -- command ...

Samples once a second, out of process: available memory (free + inactive pages), compressor
size, swap in use, OS thermal pressure level, power source and battery. Stops the command's
process group (SIGINT, then SIGTERM, then SIGKILL) when

  * available memory stays under the floor for --floor-samples consecutive samples,
  * swap in use rises more than --swap-growth-mib above the value at launch,
  * the compressor ratchets upward: its minimum over a 30 s window rises by more than
    --compressor-step-mib in four consecutive windows, or it stays more than
    --compressor-growth-mib above launch for 20 consecutive samples. Short bursts that are
    released again (seen in every short prefill) are counted and reported, not stopped,
  * thermal pressure is at or above --thermal-level for --thermal-samples consecutive samples,
  * the machine enters Low Power Mode, or the battery is under --min-battery-percent (default 25; 0 turns the floor off).
    Running on battery is allowed: there is no plug-in requirement.

Exit status: the command's own status, or 90 when the guard stopped it.
"""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

LIBC = ctypes.CDLL(None, use_errno=True)
GIB = 2 ** 30
MIB = 2 ** 20


class VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64), ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class XswUsage(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64), ("used", ctypes.c_uint64),
                ("pagesize", ctypes.c_uint32), ("encrypted", ctypes.c_uint32)]


LIBC.mach_host_self.restype = ctypes.c_uint32
HOST = LIBC.mach_host_self()
PAGE = os.sysconf("SC_PAGE_SIZE")
LIBC.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
LIBC.host_statistics64.restype = ctypes.c_int
THERMAL_TOKEN = ctypes.c_int(0)
LIBC.notify_register_check(b"com.apple.system.thermalpressurelevel", ctypes.byref(THERMAL_TOKEN))


def memory():
    stats = VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    if LIBC.host_statistics64(HOST, 4, ctypes.byref(stats), ctypes.byref(count)):
        raise OSError("host_statistics64 failed")
    swap = XswUsage()
    size = ctypes.c_size_t(ctypes.sizeof(swap))
    if LIBC.sysctlbyname(b"vm.swapusage", ctypes.byref(swap), ctypes.byref(size), None, 0):
        raise OSError("vm.swapusage failed")
    return {
        "available": (stats.free_count + stats.inactive_count) * PAGE,
        "free": stats.free_count * PAGE,
        "compressor": stats.compressor_page_count * PAGE,
        "swap_used": swap.used,
        "swapouts": stats.swapouts * PAGE,
        "swapins": stats.swapins * PAGE,
        "compressions": stats.compressions * PAGE,
    }


def thermal():
    state = ctypes.c_uint64(0)
    LIBC.notify_get_state(THERMAL_TOKEN, ctypes.byref(state))
    return int(state.value)


def power():
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        mode = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {"ac": None, "battery": None, "low_power": None}
    percent = None
    for part in batt.replace(";", " ").split():
        if part.endswith("%"):
            percent = int(part[:-1])
    low = None
    for line in mode.splitlines():
        if "lowpowermode" in line:
            low = line.split()[-1] == "1"
    return {"ac": "AC Power" in batt, "battery": percent, "low_power": low}


def stop(process):
    for signum, wait in ((signal.SIGINT, 20), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        except PermissionError:
            if process.poll() is not None:
                return                  # macOS refuses killpg on a group whose leader already exited
            raise
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--floor-gib", type=float, default=3.0)
    parser.add_argument("--floor-samples", type=int, default=3)
    parser.add_argument("--swap-growth-mib", type=float, default=512)
    parser.add_argument("--compressor-growth-mib", type=float, default=4096)
    parser.add_argument("--compressor-step-mib", type=float, default=128)
    parser.add_argument("--thermal-level", type=int, default=2)
    parser.add_argument("--thermal-samples", type=int, default=15)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--min-battery-percent", type=int, default=25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command")

    base, state = memory(), power()
    refusal = None
    if state["low_power"]:
        refusal = "Low Power Mode is on"
    elif state["ac"] is False and state["battery"] is not None and state["battery"] < args.min_battery_percent:
        refusal = f"battery under {args.min_battery_percent}%"
    elif thermal() >= args.thermal_level:
        refusal = "thermal pressure already high"
    elif base["available"] < args.floor_gib * GIB:
        refusal = f"available memory {base['available'] / GIB:.2f} GiB is already under the floor"
    with open(args.log, "w") as log:
        log.write(json.dumps({"event": "start", "command": command, "baseline": base, "power": state,
                              "thermal": thermal(), "limits": vars(args) | {"command": None},
                              "refusal": refusal, "time": time.time()}) + "\n")
        if refusal:
            print(f"guard: refused to start: {refusal}", flush=True)
            return 91
        process = subprocess.Popen(command, start_new_session=True)
        began = time.monotonic()
        low = hot = growing = high = 0
        window, last_floor = [], base["compressor"]
        floor, swap_peak, comp_peak, levels, reason = base["available"], base["swap_used"], base["compressor"], {}, None
        tick = 0
        while process.poll() is None:
            time.sleep(1.0)
            tick += 1
            now, level = memory(), thermal()
            row = {"t": round(time.monotonic() - began, 1), **now, "thermal": level}
            floor = min(floor, now["available"])
            swap_peak = max(swap_peak, now["swap_used"])
            comp_peak = max(comp_peak, now["compressor"])
            levels[level] = levels.get(level, 0) + 1
            low = low + 1 if now["available"] < args.floor_gib * GIB else 0
            hot = hot + 1 if level >= args.thermal_level else 0
            window.append(now["compressor"])
            high = high + 1 if now["compressor"] - base["compressor"] > args.compressor_growth_mib * MIB else 0
            if tick % 30 == 0:
                growing = growing + 1 if min(window) - last_floor > args.compressor_step_mib * MIB else 0
                last_floor, window = min(window), []
            if tick % 30 == 0:
                state = power()
                row["power"] = state
                if state["low_power"]:
                    reason = "Low Power Mode turned on"
                elif state["ac"] is False and state["battery"] is not None and state["battery"] < args.min_battery_percent:
                    reason = f"battery under {args.min_battery_percent}%"
            if low >= args.floor_samples:
                reason = (f"available memory under {args.floor_gib:g} GiB for {low} s "
                          f"(now {now['available'] / GIB:.2f} GiB)")
            elif now["swap_used"] - base["swap_used"] > args.swap_growth_mib * MIB:
                reason = f"swap grew {(now['swap_used'] - base['swap_used']) / MIB:.0f} MiB over launch"
            elif high >= 20:
                reason = f"compressor stayed {(now['compressor'] - base['compressor']) / MIB:.0f} MiB over launch for 20 s"
            elif growing >= 4:
                reason = "compressor floor rose in four consecutive 30 s windows (sustained compression)"
            elif hot >= args.thermal_samples:
                reason = f"thermal pressure level {level} for {hot} s"
            elif time.monotonic() - began > args.max_seconds:
                reason = f"ran longer than {args.max_seconds:g} s"
            log.write(json.dumps(row) + "\n")
            log.flush()
            if reason:
                print(f"guard: STOP: {reason}", flush=True)
                stop(process)
                break
        end = memory()
        summary = {
            "event": "end", "stopped": reason, "returncode": process.returncode,
            "seconds": round(time.monotonic() - began, 1),
            "available_floor_gib": round(floor / GIB, 2),
            "swap_baseline_mib": round(base["swap_used"] / MIB), "swap_peak_mib": round(swap_peak / MIB),
            "swap_end_mib": round(end["swap_used"] / MIB),
            "compressor_baseline_mib": round(base["compressor"] / MIB), "compressor_peak_mib": round(comp_peak / MIB),
            "swapout_mib": round((end["swapouts"] - base["swapouts"]) / MIB),
            "compressed_gib_total": round((end["compressions"] - base["compressions"]) / GIB, 1),
            "thermal_seconds": levels,
        }
        log.write(json.dumps(summary) + "\n")
        print("guard: " + json.dumps(summary), flush=True)
        return 90 if reason else process.returncode


if __name__ == "__main__":
    sys.exit(main())
