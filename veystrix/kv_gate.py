"""Real-checkpoint KV quality gate: lossy KV modes against bf16 KV on one long document.

For every mode the same document is prefilled into the generic C++ runtime (one engine per mode, one
process, run it under scripts/flashnext/real_run_guard.py):

* continuation probe: the last ``--forced`` document tokens are teacher-forced; at each position the
  greedy argmax and the log-probability of the real next token are recorded (sampler
  ``{"force": t, "probe": true, "logprob": true}``). Top-1 agreement with bf16 and the per-token NLL
  delta (as a perplexity ratio) are the overall quality numbers.
* needle probes: facts ("The access code for vault K is V.") are planted in the document; each fact sits
  in a known KV region (protected prefix, quantized middle history, recent window). For each needle a
  fork of the finished document asks the question; the free greedy answer must contain the value
  (needle accuracy), and bf16's answer is teacher-forced so the argmax agreement on the answer tokens is
  the "history-region probe" number.
* tool calls: ``--tool-prompts`` requests after the document ask the model to call one of four declared
  tools with values it must take from the document (or from the request). The answer is parsed by the
  server's own tool-call parser; valid = one call to the expected tool with well-typed arguments,
  correct = valid and the values match.
* speed and memory: decode tok/s over every free answer, MLX active memory after the prefill and the
  peak during it.

Pass bar for a mode. Every run should include the noise arm ``bf16#c1024``: bf16 again, prefilled in 1024- instead
of 2048-token chunks, i.e. the same math in another floating-point order. On this 4-bit model that alone flips
many near-tie argmaxes (Ornith 32K: raw top-1 83 %, confident top-1 95 %), so the bars are relative to it:
top-1 on the positions where bf16 is confident (p(argmax) >= 0.5) at least the noise arm's minus 2 points,
>= 99 % where bf16 is sure (p >= 0.9), middle-region probes likewise, every needle bf16 finds is found,
perplexity not measurably worse (95 % bootstrap interval of the ratio reaches 1.0) and <= 1.02, no more invalid
tool calls than bf16 and at most one more tool request lost than the noise arm lost. Without a noise arm the
absolute bar applies (top-1 >= 99 %, ppl ratio <= 1.01, at most one tool request lost).

Modes may carry a window: ``tiered@49152`` (the ``auto`` plan's choice for Ornith at 262k). With
``--no-reference`` the bf16 run is skipped (contexts whose bf16 KV does not fit); modes then get absolute
numbers and the verdict "unreferenced". ``--record`` merges the verdicts into ``~/.veystrix/kv-gate.json``
(read by kv_policy.gate_passed; a mode passes only if it passed at every recorded size) and writes the
evidence to ``~/.veystrix/parity/kv-gate-<model_type>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
         "kestrel", "lagoon", "marble", "nimbus", "obsidian", "pewter", "quartz", "raven", "saffron", "tundra")
THINK_OFF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

TOOLS = [
    {"type": "function", "function": {
        "name": "record_code", "description": "Record the access code of a vault in the audit log.",
        "parameters": {"type": "object", "properties": {
            "vault": {"type": "string", "description": "vault number"},
            "code": {"type": "string", "description": "the access code, exactly as written"}},
            "required": ["vault", "code"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write text to a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "send_message", "description": "Send a chat message to a team.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}}},
    {"type": "function", "function": {
        "name": "lookup_history", "description": "Look up the access history of a vault.",
        "parameters": {"type": "object", "properties": {
            "vault": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["vault", "limit"]}}},
]


def corpus_text(limit_chars: int) -> str:
    """Natural English engineering prose and code from the repo (docs, then Python; deterministic order)."""
    parts, total = [], 0
    paths = sorted((ROOT / "docs").rglob("*.md")) + sorted((ROOT / "veystrix").rglob("*.py"))
    for path in paths:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        parts.append(text)
        total += len(text)
        if total >= limit_chars:
            break
    return "\n\n".join(parts)[:limit_chars]


def chat_head(tokenizer, tools=TOOLS) -> list[int] | None:
    """The chat template's tokens up to the user message's content (system block with the tools)."""
    marker = "\x00DOC\x00"
    try:
        text = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tools=tools,
                                             tokenize=False, add_generation_prompt=False)
    except Exception:
        return None
    if marker not in text:
        return None
    return tokenizer.encode(text.split(marker, 1)[0] + "Read the following document carefully.\n\n",
                            add_special_tokens=False)


def build(tokenizer, n_tokens: int, needles: int, window: int, prefix: int, seed: int = 7, head=None):
    """Token ids of a user message holding ~n_tokens of document with planted facts, and the facts."""
    rng = random.Random(seed)
    head = list(head) if head else tokenizer.encode("<|im_start|>user\nRead the following document carefully.\n\n")
    body = tokenizer.encode(corpus_text(n_tokens * 6))[: n_tokens - len(head)]
    n = len(head) + len(body)
    # Positions: one in the protected prefix, most in the middle (quantized history), one in the window.
    spots = [min(prefix // 2, n // 10)]
    lo, hi = prefix + 256, max(prefix + 512, n - window - 256)
    spots += [int(lo + (hi - lo) * (i + 0.5) / max(1, needles - 2)) for i in range(max(0, needles - 2))]
    spots.append(n - window // 2)
    facts, ids, cur = [], list(head), 0
    for pos in sorted(set(p for p in spots if len(head) < p < n)):
        cut = pos - len(head)
        ids += body[cur:cut]
        cur = cut
        f = {"key": f"{rng.randrange(1000, 9999)}", "value": f"{rng.choice(WORDS).upper()}-{rng.randrange(10, 99)}",
             "at": len(ids)}
        ids += tokenizer.encode(f"\nThe access code for vault {f['key']} is {f['value']}.\n")
        facts.append(f)
    ids += body[cur:]
    for f in facts:
        f["region"] = "prefix" if f["at"] < prefix else ("recent" if f["at"] >= len(ids) - window else "middle")
    return ids, facts


def question(tokenizer, key: str) -> list[int]:
    return tokenizer.encode(f"\n\nQuestion: what is the access code for vault {key}? Reply with the code only."
                            + THINK_OFF)


def tool_tasks(facts, count: int) -> list[dict]:
    """``count`` tool requests cycling over the four tools and the planted facts."""
    tasks = []
    for i in range(count):
        f = facts[i % len(facts)]
        key, value = f["key"], f["value"]
        kind = i % 4
        if kind == 0:
            tasks.append({"tool": "record_code", "expect": {"vault": key, "code": value},
                          "prompt": f"Call the record_code tool to record the access code of vault {key}, "
                                    "exactly as the document gives it."})
        elif kind == 1:
            tasks.append({"tool": "write_file", "expect": {"path": f"vault_{key}.txt", "content~": value},
                          "prompt": f"Use the write_file tool to save the access code of vault {key} (only the "
                                    f"code) into the file vault_{key}.txt."})
        elif kind == 2:
            tasks.append({"tool": "send_message", "expect": {"to": "security", "text~": value},
                          "prompt": f"Send the access code of vault {key} to the team 'security' with the "
                                    "send_message tool."})
        else:
            limit = 3 + i % 7
            tasks.append({"tool": "lookup_history", "expect": {"vault": key, "limit": limit},
                          "prompt": f"Call lookup_history for vault {key} with a limit of {limit} entries."})
    return tasks


def check_tool(text: str, task: dict) -> dict:
    """Parse with the server's parser; valid = one call to the expected tool, well-typed arguments."""
    from veystrix.server import _parse_tool_calls
    try:
        _, calls = _parse_tool_calls(text, TOOLS)
    except Exception as exc:  # the parser refuses malformed arguments
        return {"valid": False, "correct": False, "why": f"parser: {str(exc)[:120]}"}
    if not calls:
        return {"valid": False, "correct": False, "why": "no tool call"}
    call = calls[0]
    try:
        args = json.loads(call["arguments"])
    except (TypeError, ValueError):
        return {"valid": False, "correct": False, "why": "arguments are not JSON"}
    schema = next(t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == task["tool"])
    types = {"string": str, "integer": int}
    valid = (call["name"] == task["tool"] and isinstance(args, dict)
             and all(k in args for k in schema["required"])
             and all(isinstance(args[k], types[v["type"]]) and not isinstance(args[k], bool)
                     for k, v in schema["properties"].items() if k in args))
    correct = valid
    for key, want in task["expect"].items():
        if key.endswith("~"):
            correct = correct and str(want) in str(args.get(key[:-1], ""))
        else:
            correct = correct and args.get(key) == want
    return {"valid": bool(valid), "correct": bool(correct), "name": call["name"], "args": args}


def parse_mode(spec: str) -> tuple[str, int | None]:
    """``name[@window][:hK/V]``, e.g. ``tiered@49152`` or ``tiered@32768:h8/4`` (bf16 window, K8/V4 history)."""
    name, _, win = spec.split("#", 1)[0].split(":", 1)[0].partition("@")
    return name, (int(win) if win else None)


def parse_chunk(spec: str) -> int:
    """``bf16#c1024``: the same mode prefilled in 1024-token chunks. With bf16 this is the gate's noise floor
    (identical math, another floating-point reduction order), so a lossy mode is judged against it."""
    _, _, extra = spec.partition("#")
    return int(extra[1:]) if extra.startswith("c") else 2048


def parse_bits(spec: str) -> tuple[int, int] | None:
    _, _, bits = spec.split("#", 1)[0].partition(":")
    if not bits.startswith("h"):
        return None
    k, _, v = bits[1:].partition("/")
    return int(k), int(v)


def _lp(digest: str, field: str = "lp") -> float | None:
    """A value from the probe's "lp=<log p(forced)> top=<log p(argmax)>" digest (None for other digests)."""
    if not digest or not digest.startswith("lp="):
        return None
    for part in digest.split():
        name, _, value = part.partition("=")
        if name == field:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _generate(b, session, n, stop):
    out, t0 = [], time.perf_counter()
    for _ in range(n):
        t = b.step(session, '{"greedy": true}')
        out.append(t)
        if t in stop:
            break
    return out, time.perf_counter() - t0


def run_mode(lib, ckpt, recipe, mode, window, prefix, ids, facts, forced, answer_tokens, tokenizer,
             ref_answers=None, tasks=None, tool_tokens=192):
    from veystrix.port_check import Cpp
    name, own_window = parse_mode(mode)
    window = own_window or window
    try:
        import mlx.core as mx
    except Exception:  # memory numbers are optional
        mx = None
    peak = {"v": 0}
    done = threading.Event()

    def sample():
        while not done.wait(0.05):
            peak["v"] = max(peak["v"], mx.get_active_memory())

    t0 = time.time()
    chunk = parse_chunk(mode)
    cpp = Cpp(lib, ckpt, recipe, name, prefill_chunk=chunk,
              window=None if name in ("bf16", "int8", "k8v4", "int4") else window,
              prefix=prefix if name in ("tiered4+", "tiered-k8v4") else None, hist_bits=parse_bits(mode))
    b = cpp.b
    base = mx.get_active_memory() if mx else 0
    watcher = threading.Thread(target=sample, daemon=True) if mx else None
    if watcher:
        watcher.start()
    s = cpp.session()
    stop = {t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if t is not None}
    body, tail = ids[:-forced], ids[-forced:]
    b.prefill(s, body[:chunk])
    for at in range(chunk, len(body), chunk):
        b.prefill_ex(s, body[at:at + chunk], 1)
    prefill_s = time.time() - t0
    argmax, nll, top = [], [], []
    for t in tail:
        argmax.append(b.step(s, json.dumps({"force": t, "probe": True, "logprob": True})))
        digest = b.last_digest(s)
        lp = _lp(digest)
        nll.append(None if lp is None else -lp)
        top.append(_lp(digest, "top"))
    after = mx.get_active_memory() if mx else 0
    out = {"mode": mode, "prefill_seconds": round(prefill_s, 1), "continuation_argmax": argmax,
           "continuation_nll": nll, "continuation_top_lp": top, "needles": [], "tools": []}
    gen_tokens, gen_seconds = 0, 0.0
    for i, f in enumerate(facts):
        fork = cpp.session()
        b.fork(s, fork)
        b.prefill_ex(fork, question(tokenizer, f["key"]), 1)
        if ref_answers is None:
            ans, dt = _generate(b, fork, answer_tokens, stop)
            gen_tokens, gen_seconds = gen_tokens + len(ans), gen_seconds + dt
            probe = ans
        else:
            probe = [b.step(fork, json.dumps({"force": t, "probe": True})) for t in ref_answers[i]]
            free = cpp.session()
            b.fork(s, free)
            b.prefill_ex(free, question(tokenizer, f["key"]), 1)
            ans, dt = _generate(b, free, answer_tokens, stop)
            gen_tokens, gen_seconds = gen_tokens + len(ans), gen_seconds + dt
            b.close_session(free)
        b.close_session(fork)
        text = tokenizer.decode(ans)
        out["needles"].append({**f, "answer": text, "found": f["value"] in text, "answer_ids": ans,
                               "probe_argmax": probe})
    for task in tasks or []:
        fork = cpp.session()
        b.fork(s, fork)
        b.prefill_ex(fork, tokenizer.encode("\n\n" + task["prompt"] + THINK_OFF), 1)
        ans, dt = _generate(b, fork, tool_tokens, stop)
        gen_tokens, gen_seconds = gen_tokens + len(ans), gen_seconds + dt
        b.close_session(fork)
        text = tokenizer.decode(ans)
        out["tools"].append({"tool": task["tool"], "text": text, **check_tool(text, task),
                             "truncated": len(ans) >= tool_tokens and not (set(ans[-1:]) & stop)})
    done.set()
    stats = b.stats(cpp.engine)
    out["kv"] = (b.capabilities(cpp.engine) or {}).get("kv")
    out["stats"] = {k: stats.get(k) for k in ("expert_bytes_read", "kv_values_per_token", "resident_bytes")
                    if k in stats}
    out["decode_tok_s"] = round(gen_tokens / gen_seconds, 2) if gen_seconds else None
    out["decode_tokens"] = gen_tokens
    if mx:
        gib = 1024 ** 3
        out["memory_gib"] = {"engine_loaded": round(base / gib, 2), "after_prefill": round(after / gib, 2),
                             "peak": round(max(peak["v"], after) / gib, 2),
                             "context_state": round((after - base) / gib, 2)}
    out["seconds"] = round(time.time() - t0, 1)
    b.close_session(s)
    cpp.close()
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def absolute(got) -> dict:
    nll = _mean(got.get("continuation_nll", []))
    tools = got.get("tools", [])
    return {"nll": None if nll is None else round(nll, 5), "ppl": None if nll is None else round(math.exp(nll), 4),
            "needles_found": sum(n["found"] for n in got["needles"]), "needles": len(got["needles"]),
            "tools_valid": sum(t["valid"] for t in tools), "tools_correct": sum(t["correct"] for t in tools),
            "tools": len(tools), "decode_tok_s": got.get("decode_tok_s"), "memory_gib": got.get("memory_gib"),
            "prefill_seconds": got.get("prefill_seconds")}


def score(ref, got):
    cont = sum(a == b for a, b in zip(ref["continuation_argmax"], got["continuation_argmax"]))
    total = len(ref["continuation_argmax"])
    mid_hit = mid_n = all_hit = all_n = 0
    for r, g in zip(ref["needles"], got["needles"]):
        pairs = list(zip(r["answer_ids"], g["probe_argmax"]))
        hit = sum(a == b for a, b in pairs)
        all_hit += hit
        all_n += len(pairs)
        if r["region"] == "middle":
            mid_hit += hit
            mid_n += len(pairs)
    # Raw top-1 on a high-entropy document is dominated by near-ties (an argmax flips between two tokens of
    # almost equal probability under any perturbation). Agreement where bf16 is confident (p(top) >= 0.5) is
    # the part a KV error can really damage; both are reported, the confident one is gated.
    conf = [(a, b) for a, b, lp in zip(ref["continuation_argmax"], got["continuation_argmax"],
                                       ref.get("continuation_top_lp") or [])
            if lp is not None and lp >= math.log(0.5)]
    out = {"top1_continuation": cont / max(1, total), "continuation_tokens": total,
           "top1_confident": (sum(a == b for a, b in conf) / len(conf)) if conf else None,
           "confident_tokens": len(conf),
           "top1_probes": all_hit / max(1, all_n), "top1_middle_probes": mid_hit / max(1, mid_n),
           "middle_probe_tokens": mid_n,
           "needles_found": sum(n["found"] for n in got["needles"]), "needles": len(got["needles"]),
           "needles_lost": [g["key"] for r, g in zip(ref["needles"], got["needles"]) if r["found"] and not g["found"]]}
    sure = [(a, b) for a, b, lp in zip(ref["continuation_argmax"], got["continuation_argmax"],
                                       ref.get("continuation_top_lp") or [])
            if lp is not None and lp >= math.log(0.9)]
    out["top1_p90"] = (sum(a == b for a, b in sure) / len(sure)) if sure else None
    out["p90_tokens"] = len(sure)
    ref_nll, got_nll = _mean(ref.get("continuation_nll", [])), _mean(got.get("continuation_nll", []))
    if ref_nll is not None and got_nll is not None:
        out["delta_nll"] = round(got_nll - ref_nll, 6)
        out["ppl_ratio"] = round(math.exp(got_nll - ref_nll), 5)
        lo, hi = ppl_ratio_ci(ref["continuation_nll"], got["continuation_nll"])
        out["ppl_ratio_ci95"] = [round(lo, 5), round(hi, 5)]
    rt, gt = ref.get("tools", []), got.get("tools", [])
    if gt:
        # An answer cut by the token cap is unfinished, not wrong (the first runs capped at 96 tokens and a
        # longer phrasing of a correct call was scored invalid): it is excluded from the loss counts.
        def cut(t):
            return t.get("truncated", not t["text"].rstrip().endswith("<|im_end|>"))
        lost = [(a, b) for a, b in zip(rt, gt) if a["correct"] and not b["correct"] and not cut(b)]
        out.update(tools=len(gt), tools_valid=sum(t["valid"] for t in gt), tools_correct=sum(t["correct"] for t in gt),
                   ref_tools_valid=sum(t["valid"] for t in rt), ref_tools_correct=sum(t["correct"] for t in rt),
                   tools_truncated=sum(cut(t) for t in gt), ref_tools_truncated=sum(cut(t) for t in rt),
                   tools_regressed=len(lost),
                   # a well-formed call with a wrong value: the failure an agent cannot see
                   tools_value_errors=sum(b["valid"] for _, b in lost),
                   tools_gained=sum(b["correct"] and not a["correct"] for a, b in zip(rt, gt)),
                   tools_same_text=sum(a["text"] == b["text"] for a, b in zip(rt, gt)))
    return out


def ppl_ratio_ci(ref_nll, got_nll, samples: int = 2000, seed: int = 11) -> tuple[float, float]:
    """95 % paired-bootstrap interval of exp(mean NLL delta): the per-token deltas of a short continuation are
    wide (a perturbed model moves single tokens by a nat either way), so the ratio needs its error bar."""
    deltas = [g - r for r, g in zip(ref_nll, got_nll) if r is not None and g is not None]
    if not deltas:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return math.exp(means[int(0.025 * samples)]), math.exp(means[int(0.975 * samples) - 1])


def is_noise_arm(mode: str) -> bool:
    return parse_mode(mode)[0] == "bf16" and "#" in mode


def verdict(sc, noise=None) -> str:
    """Pass/fail of one lossy mode against bf16.

    With a noise arm (``bf16#c1024``: the same bf16 math in another floating-point order) the agreement bars
    are relative to it: on Ornith at 32K that arm alone moves raw top-1 to 83 % and confident top-1 to 95 %
    (results/cpp/kv-gate/orn-32k.json), so an absolute 99 % bar would fail bf16 against itself. A lossy mode
    passes when it is no further from bf16 than that noise plus a small margin, loses no needle, does not
    raise perplexity measurably, and keeps its tool calls."""
    top1 = sc["top1_confident"] if sc.get("top1_confident") is not None else sc["top1_continuation"]
    ratio = sc.get("ppl_ratio", 1.0)
    lo = (sc.get("ppl_ratio_ci95") or [ratio])[0]
    if noise is None:
        ok = (top1 >= 0.99 and (sc["middle_probe_tokens"] == 0 or sc["top1_middle_probes"] >= 0.98)
              and not sc["needles_lost"] and ratio <= 1.01)
        regress_cap = 1
    else:
        n_top1 = noise["top1_confident"] if noise.get("top1_confident") is not None else noise["top1_continuation"]
        n_mid = noise["top1_middle_probes"] if noise["middle_probe_tokens"] else 1.0
        ok = (top1 >= n_top1 - NOISE_MARGIN
              and (sc.get("top1_p90") is None or sc["top1_p90"] >= 0.99)
              and (sc["middle_probe_tokens"] == 0 or sc["top1_middle_probes"] >= n_mid - NOISE_MARGIN)
              and not sc["needles_lost"]
              and lo <= 1.0 and ratio <= 1.02)          # not measurably worse, and small even if it were
        regress_cap = max(1, noise.get("tools_regressed", 0) + 1)
    if "tools" in sc:
        # bf16 itself is not perfect (it answers a few requests in prose), so the bar is relative and per item:
        # no fewer valid calls than bf16 (an unfinished answer may still become one), at most one request bf16 got
        # right lost beyond the noise arm's losses, and at most one wrong-valued call more than the noise arm.
        valid = sc["tools_valid"] + sc.get("tools_truncated", 0)
        value_cap = (noise.get("tools_value_errors", 0) if noise else 0) + 1
        ok = (ok and valid >= sc.get("ref_tools_valid", sc["tools"]) and sc.get("tools_regressed", 0) <= regress_cap
              and sc.get("tools_value_errors", 0) <= value_cap)
    return "pass" if ok else "fail"


NOISE_MARGIN = 0.02


def record(model_type: str, tokens: int, result: dict) -> None:
    from veystrix.kv_policy import gate_record_path
    path = gate_record_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    entry = data.setdefault(model_type, {})
    sizes = entry.setdefault("sizes", {})
    sizes[str(tokens)] = result.get("verdicts", {})
    for mode in [m for m in entry if is_noise_arm(m)]:
        entry.pop(mode)                            # a noise arm is the yardstick, never a candidate
    for mode in {m for v in sizes.values() for m, x in v.items() if x != "noise-floor"}:
        seen = [v[mode] for v in sizes.values() if mode in v]
        if any(x == "fail" for x in seen):
            entry[mode] = "fail"
        elif any(x == "pass" for x in seen):
            entry[mode] = "pass"
        else:
            entry[mode] = seen[-1]
    entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    evidence = path.parent / "parity" / f"kv-gate-{model_type}.json"
    try:
        ev = json.loads(evidence.read_text())
    except (OSError, ValueError):
        ev = {}
    ev[str(tokens)] = {"at": entry["at"], "window": result["window"], "prefix": result["prefix"],
                       "verdicts": result.get("verdicts", {}),
                       "scores": {m: r.get("score") for m, r in result["modes"].items()},
                       "absolute": {m: r.get("absolute") for m, r in result["modes"].items()}}
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(ev, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lib")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--forced", type=int, default=512)
    ap.add_argument("--answer-tokens", type=int, default=12)
    ap.add_argument("--tool-prompts", type=int, default=24)
    ap.add_argument("--modes", default="tiered4+,tiered-k8v4")
    ap.add_argument("--no-reference", action="store_true", help="skip bf16 (its KV does not fit this context)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", action="store_true", help="merge the verdicts into ~/.veystrix/kv-gate.json")
    ap.add_argument("--rescore", action="store_true", help="recompute scores and verdicts from --out, no model run")
    ap.add_argument("--append", action="store_true",
                    help="add --modes to an existing --out (same document; its bf16 reference is reused)")
    args = ap.parse_args(argv)
    if args.rescore:
        result = json.loads(Path(args.out).read_text())
        ref = result["modes"].get("bf16")
        verdicts = {}
        noise = None
        order = sorted(result["modes"], key=lambda m: not is_noise_arm(m))   # the noise arm first
        for mode in order:
            got = result["modes"][mode]
            got["absolute"] = absolute(got)
            if mode == "bf16":
                continue
            if ref is None:
                verdicts[mode] = "unreferenced"
                continue
            got["score"] = score(ref, got)
            if is_noise_arm(mode):
                noise = got["score"]
                verdicts[mode] = "noise-floor"
            else:
                verdicts[mode] = verdict(got["score"], noise)
            print(f"{mode}: {verdicts[mode].upper()} {json.dumps(got['score'])}", flush=True)
        result["verdicts"] = verdicts
        Path(args.out).write_text(json.dumps(result))
        if args.record:
            record(result.get("model_type", "unknown"), result["tokens"], result)
        return 0 if all(v in ("pass", "unreferenced", "noise-floor") for v in verdicts.values()) else 1
    from transformers import AutoTokenizer
    from veystrix.recipe import ensure_recipe
    import os
    ckpt = Path(args.checkpoint).expanduser()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    recipe = ensure_recipe(ckpt)
    model_type = json.loads(Path(recipe).read_text()).get("model_type", "unknown")
    lib = args.lib or os.environ.get("VEYSTRIX_CPP_LIB") or str(ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib")
    ids, facts = build(tok, args.tokens, args.needles, args.window, args.prefix, head=chat_head(tok))
    tasks = tool_tasks(facts, args.tool_prompts)
    print(f"{model_type}: document {len(ids)} tokens, needles at {[(f['at'], f['region']) for f in facts]}, "
          f"{len(tasks)} tool prompts", flush=True)
    result = {"model_type": model_type, "tokens": len(ids), "window": args.window, "prefix": args.prefix,
              "facts": facts, "modes": {}}
    if args.append:
        # Same document (deterministic build), so the saved bf16 reference and earlier modes are reused.
        old = json.loads(Path(args.out).read_text())
        if old.get("tokens") != len(ids) or old.get("facts") != facts:
            raise SystemExit(f"--append: {args.out} was made from another document")
        result = old
    ref = result["modes"].get("bf16")
    if ref is None and not args.no_reference:
        ref = run_mode(lib, ckpt, recipe, "bf16", args.window, args.prefix, ids, facts, args.forced,
                       args.answer_tokens, tok, tasks=tasks)
        ref["absolute"] = absolute(ref)
        result["modes"]["bf16"] = ref
        print(f"bf16: {json.dumps(ref['absolute'])}", flush=True)
        Path(args.out).write_text(json.dumps(result))
    verdicts = dict(result.get("verdicts") or {})
    modes = sorted([m for m in args.modes.split(",") if m], key=lambda m: not is_noise_arm(m))
    noise_mode = next((m for m in result["modes"] if is_noise_arm(m)), None)
    noise = result["modes"][noise_mode].get("score") if noise_mode else None
    for mode in modes:
        got = run_mode(lib, ckpt, recipe, mode, args.window, args.prefix, ids, facts, args.forced,
                       args.answer_tokens, tok, tasks=tasks,
                       ref_answers=None if ref is None else [n["answer_ids"] for n in ref["needles"]])
        got["absolute"] = absolute(got)
        if ref is not None:
            got["score"] = sc = score(ref, got)
            if is_noise_arm(mode):
                noise, verdicts[mode] = sc, "noise-floor"
            else:
                verdicts[mode] = verdict(sc, noise)
        else:
            sc = None
            verdicts[mode] = "unreferenced"
        result["modes"][mode] = got
        print(f"{mode}: {verdicts[mode].upper()} {json.dumps(sc) if sc else ''} {json.dumps(got['absolute'])} "
              f"kv {json.dumps(got['kv'])}", flush=True)
        Path(args.out).write_text(json.dumps(result))
    result["verdicts"] = verdicts
    Path(args.out).write_text(json.dumps(result))
    if args.record:
        record(model_type, len(ids), result)
    return 0 if all(v in ("pass", "unreferenced", "noise-floor") for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
