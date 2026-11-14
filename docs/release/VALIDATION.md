# Validation checklist and last recorded status (2026-11-10, qwen3.8-flash-next)

Run before every release. "Ran" means executed on this repo state; "NOT RUN" means not executed and **not** claimed.

| Check | Command | Status |
|---|---|---|
| Shared runtime unit tests | `python -m unittest discover -s veystrix/tests -t . -p 'test_*.py'` | Ran: 791 tests OK (8 skipped, incl. the gated Hermes e2e) |
| Flash-Next / Qwen27B / K2 / Bonsai unit tests | the four `models/*/tests/unit` discoveries | Ran: 1818 (3 skipped) / 22 / 58 / 175 (5 skipped), all OK |
| Swift app logic tests | `swift test --package-path apps/macos/Veystrix` | 40 pass (swift-testing) |
| App builds (dev, ad-hoc) and launches with engine stopped | `packaging/macos/build_app.sh` then run | Ran: opens, no child process |
| Hermes real-client e2e vs mock | `VEYSTRIX_HERMES_E2E=1 python -m unittest veystrix.tests.test_hermes_integration` | 25/25 pass, zero egress |
| Release packaging tests | `python -m unittest veystrix.tests.test_release_packaging` | pass |
| Secret scan (tree + history) | `python scripts/security/secret_scan.py [--history]` | 0 hits |
| Public exposure audit | `python scripts/security/public_exposure_audit.py --public DIR --core DIR` | run when the public repo exists |
| Engine bundle build + import smoke | `packaging/engine/build_engine_bundle.sh` | Ran (scratch): imports + `--help` ok |
| Developer ID signing, notarization, stapling | `packaging/macos/make_release.sh --notarize` | **NOT RUN — no credentials** |
| Sparkle key generation, signed appcast, update install | `make_appcast.sh` | **NOT RUN — no credentials** |
| Real-checkpoint engine launch / chat / benchmark | — | **NOT RUN — OWNER TRAVELLING** |
| Layer-major long-lived server safety; 16K+ throughput; logit-level equivalence | — | **NOT RUN — OWNER TRAVELLING** |
| Hardware validation of the signed app (entitlements for MLX/Metal) | — | **NOT RUN** |
| Interactive chat flow test; `test_safe_memory_preflight` | — | **NOT RUN** |

## Status on 2026-11-14 (branch `unlimited-generation`, local commits only)

The table above is the 10 November record. These checks were run again on 14 November; details and evidence paths are in [the handover](../performance/handoff-2026-11-14.md).

| Check | Status on 14 November |
|---|---|
| Shared runtime unit tests | Ran: 933 OK (11 skipped) |
| Flash-Next unit tests | Ran: 2253 OK (3 skipped) |
| Nemotron unit tests | Ran: 171 OK (8 skipped) |
| Swift app logic tests | Ran: 71 pass |
| Engine bundle build + import smoke | Ran: passed (`dist/engine-20261114`). A bundle built between `a86802d` and `b0d786d` could not start a model |
| App builds (ad-hoc, free channel) and launches with engine stopped | Ran: opens, starts no engine, quits, no crash report |
| Real-checkpoint launch and chat through the packaged engine | Ran under the run guard with the owner's saved settings: Nemotron and Qwen each answered one plain and one streaming chat completion; Qwen also with the low-memory prefill setting |
| Interactive chat flow in the app window | **NOT RUN** (nobody clicked through the app) |
| Installed app in `/Applications` | **NOT UPDATED**: still the 13 November build |
| Developer ID signing, notarization, Sparkle appcast | **NOT RUN — no credentials** |
| 20 tok/s release gate | **OPEN on both models** (Flash-Next 2.25–2.86 tok/s, Nemotron 5.3–6.5 tok/s) |
