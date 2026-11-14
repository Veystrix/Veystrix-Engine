"""Opt-in deferral of automatic cyclic collection on the inference thread."""
import gc
import os


def suspend():
    enabled = os.environ.get("VEYSTRIX_DEFER_DECODE_GC") == "1" and gc.isenabled()
    if enabled:
        gc.disable()
    return enabled


def restore(suspended):
    if suspended:
        gc.enable()
