# Internal app map — Veystrix macOS app, launch path, packaging, protection

Date: 2026-11-17. Branch examined: `neural-engine` (33 commits ahead of `main`, 0 behind).
Scope: read-only codebase research. No files modified, no benchmarks run, no builds performed.

> Performance caveat: this note does **not** assess ANE performance. It reports only what the code
> and docs in the repository say. The one direct ANE investigation in the repo is
> `experiments/ane/REPORT.md`, whose own status line is "evaluated and rejected"; its measured
> claims and its stated limitations are quoted below, not re-derived here.

---

## Executive summary

- The app is a thin SwiftUI front end. It never loads a model itself. `AppSettings` is turned into a
  `LaunchPlan` (process, args, env), and `SystemEngineLauncher` starts a Python engine process that
  owns the model and all inference.
- The clean seam for "use ANE" is therefore **not** in Swift inference code (there is none); it is a
  new setting threaded from `AppSettings` → `SettingsView` → `LaunchPlan` (one arg or env var), plus
  a consumer on the Python engine side and a small set of test/doc/packaging updates.
- Shipping constraints are dominated by code protection and licensing: the engine bundle is compiled,
  stripped of source, and encrypted; integrity is manifested by SHA-256 over `engine/lib/` and a few
  text resources; there is an explicit, repeated policy that **no model weights / checkpoints ship** in
  the repo, app or release artifacts. A Core ML `.mlmodelc` containing weights would conflict with
  that policy and would need a legal/notice change, not just a build-script change.
- Public release state: **no public release has been made** (free channel, ad-hoc signed, not
  notarized), while `CHANGELOG.md` labels 1.0.0 a "public release candidate".
- The ANE route has already been investigated and explicitly closed in `experiments/ane/REPORT.md`.

---

## 1. App structure and settings → engine flow

### Swift sources (exact paths)

`apps/macos/Veystrix/Sources/Veystrix/` (UI / executable target):

| File | Role |
|---|---|
| `VeystrixApp.swift` | App entry point |
| `AppComponents.swift` | Shared view components (e.g. `PageHeader`) |
| `AppModel.swift` | Ties settings, controller, checkpoint report and updater together; gates engine start |
| `ContentView.swift`, `DashboardView.swift`, `ChatView.swift`, `ModelView.swift`, `LogsView.swift`, `StatisticsView.swift`, `HarnessView.swift`, `WorkflowPanel.swift` | Feature views |
| `SettingsView.swift` | All user preferences, including power/performance/engine |
| `CacheSettings.swift` | Cache-quota UI |
| `UpdateController.swift` | Sparkle updater wrapper |

`apps/macos/Veystrix/Sources/VeystrixKit/` (logic library target, UI-free):

`APIKeyStore.swift`, `BundleIntegrity.swift`, `CacheManagement.swift`, `ChatClient.swift`,
`ChatHistoryCipher.swift`, `CheckpointValidator.swift`, `DecodeAdvisor.swift`, `Diagnostics.swift`,
`EngineController.swift`, `EngineModels.swift`, `EngineProcess.swift`, `HarnessProfiles.swift`,
`HealthClient.swift`, `LaunchPlan.swift`, `LogStore.swift`, `MemoryProbe.swift`, `ModelIdentity.swift`,
`Paths.swift`, `PersistedState.swift`, `Sanitizer.swift`, `Settings.swift`, `UsageStats.swift`,
`WorkflowClient.swift`.

Package manifest: `apps/macos/Veystrix/Package.swift` — macOS 14 targets Swift 6 tools, Swift language
mode v5, Sparkle 2.10.1 pinned; `VeystrixKit` is the tested logic library.

### What the named files do

- **`VeystrixKit/Settings.swift`**
  - Enums: `DecodePool` (lines 6–31), `ModelFamily` (34–48), `FlashNextKVCache` (50–60),
    `DecodeStrategy` (62–73), `NemotronKVCache` (75–83), `PerformanceOptions` (85–110).
  - `AppSettings` (113–255): all persisted preferences (port, checkpoint paths, selected model,
    decode pool, swap limit, exact speed, bounded prefill, power guards, context, developer repo path…).
    `CodingKeys` 185–191, custom `init(from:)` 193–232 (default-fill and migration),
    `validationProblems()` 235–254. Schema version 2 (114), swap range 1…1024 (115).
  - `SettingsStore` (259–291): atomic, owner-only (0600) JSON load/save; corrupted file is moved to
    `settings.json.damaged` and defaults are used.
  - Per-model computed selectors: `activeCheckpointPath` 164–167, `activePerformance` 169–172,
    `activeContextTokens` 175–178, `activeContextLimit` 181–183.
- **`Veystrix/SettingsView.swift`** (142 lines): SwiftUI `Form`, sections Engine (15–46),
  Performance (47–66), Power (67–72), Loading (73–77), Cache (78), Local API key (79–87),
  Updates (88–95), Privacy (96–100), Developer (101–110), Storage (111–114). Each toggle binds
  directly to `model.settings.*`; persistence happens via `AppModel.settings.didSet` → `persist()`.
- **`VeystrixKit/LaunchPlan.swift`**: pure value type describing the engine process; `make(...)`
  (38–151) validates settings/checkpoint/runtime and builds args + env. `Source` (12–17) is
  `.bundled(engineRoot:)` or `.developer(repoRoot:)`. `inheritedKeys` (36) is the only environment
  allowlist. See §2.
- **`VeystrixKit/EngineController.swift`** (`@MainActor`): owns lifecycle; nothing runs at app launch.
  `start(...)` (61–105) requires `userInitiated: true`, refuses low battery / Low Power Mode, calls
  `LaunchPlan.make`, probes the port, then launches. Monitoring/health 112–159, stop escalation 189–217,
  restart 221–224, quit shutdown 227–236, external-engine detection 240–253.
- **`VeystrixKit/EngineProcess.swift`**: `EngineProcess` / `EngineProcessLauncher` protocols (4–16),
  `SystemEngineProcess` (18–25), and `SystemEngineLauncher` (27–71) which sets
  `executableURL/arguments/environment/currentDirectoryURL` from the plan (33–36), wires pipes and
  exit handler (38–52), and starts an optional `caffeinate -ims -w <pid>` child (54–68).
  `EngineLocator.resolve` (74–87) prefers `Contents/Resources/engine`, then
  `developerRepoPath` / `VEYSTRIX_REPO`.

### How `AppSettings` reaches the Python process

1. `AppModel.startNow()` (`AppModel.swift:119–129`) calls
   `engine.start(settings:, checkpoint:, apiKey:, userInitiated: true, source: EngineLocator.resolve(...))`.
2. `EngineController.start` (`EngineController.swift:61–105`) → `LaunchPlan.make(...)`
   (`LaunchPlan.swift:38–151`).
3. `LaunchPlan.make` converts settings into **both** command-line arguments (79–92) and environment
   variables (94–146). Notably: API key is env-only (`VEYSTRIX_API_KEY`, 144), and the server key is
   never on the command line (`VEYSTRIX_SERVER_API_KEY`, 145).
4. `SystemEngineLauncher.launch` (`EngineProcess.swift:29–70`) applies the plan to a `Process`.
5. The engine entry point is `-m veystrix.model_router` (`LaunchPlan.swift:79`).

So there is no in-process Swift ML call path at all; all execution decisions live in Python.

---

## 2. LaunchPlan — full argument/env surface, model selection, safety vars

### Launch arguments (`LaunchPlan.swift:79–92`)

| Arg | Source line | Notes |
|---|---|---|
| `-m veystrix.model_router` | 79 | Engine entry point |
| `--host 127.0.0.1` | 80 | Loopback only, always |
| `--port <settings.port>` | 81 | |
| `--pool-gb <decodePool.rawValue>` | 78, 82 | Qwen only; Nemotron sizes its own pool (comment 76–77) |
| `--checkpoint <activeCheckpointPath>` | 83 | |
| `--model <selectedModel.rawValue>` | 84 | `flashnext` or `nemotron35-lightning` |
| `--flashnext-kv-cache <rawValue>` | 85 | `bf16` / `tiered4` |
| `--flashnext-memory-policy absolute-swap-only` | 86 | Hard-coded |
| `--flashnext-speed` | 87 | Optional |
| `--flashnext-exact-speed` | 88 | Optional (on by default) |
| `--flashnext-bounded-prefill` | 89 | Optional, mutually exclusive with speed (Settings validation 247–249) |
| `--overnight` | 90 | When `supervisedRestarts` |
| `--prefix-dir <cacheDirectory>` | 92 | Only if non-empty |

### Environment variables (`LaunchPlan.swift:94–146`)

Inherited only from `inheritedKeys` (36): `HOME, USER, LOGNAME, TMPDIR, LANG, LC_ALL, PATH, SHELL,
VEYSTRIX_MODEL_ROOT` (95). Everything else is set explicitly:

| Variable | Line | Value / meaning |
|---|---|---|
| `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE` | 96–97 | `1` |
| `VEYSTRIX_MIN_BATTERY_PERCENT` | 98 | `minimumBatteryPercent` |
| `VEYSTRIX_ALLOW_LOW_POWER` | 99 | `1`/`0` |
| `VEYSTRIX_INVALID_TOOL_CALLS` | 103 | Hard-coded `"pass"` (safety/robustness; comment 100–102) |
| `VEYSTRIX_MODEL_HANDOFF` | 107 | Hard-coded `"auto"` (comment 104–106) |
| `FLASHNEXT_SWAP_CEILING_MIB` | 108 | `swapLimitGiB * 1024` |
| `VEYSTRIX_PORT` | 109 | |
| `VEYSTRIX_USAGE_LOG` | 111 | Counts-only lifetime log path |
| `VEYSTRIX_CACHE_GB` | 112 | |
| `VEYSTRIX_NEMOTRON_CONTEXT` | 113 | |
| `VEYSTRIX_NEMOTRON_EXTENDED_CONTEXT` | 116 | `1` if context > 262,144 (LCT-05 strict gate otherwise off) |
| `VEYSTRIX_NEMOTRON_KV` | 117 | |
| `VEYSTRIX_NEMOTRON_DENSE_GEMM` | 121 | Tied to `flashNextExactSpeed` |
| Per-model loop (122–128): `FLASHNEXT_*` / `VEYSTRIX_NEMOTRON_*` | 123–127 | `_PROMPT_LOOKUP`, `_PREFILL_SEED_DECODE`, `_PREFIX_ASYNC_WRITE`, `_METAL_FMA_DEQUANT`, `_MLX_AUTO_BUFFER_CAPS` |
| `VEYSTRIX_FLASHNEXT_NATIVE_MTP`, `VEYSTRIX_NEMOTRON_NATIVE_MTP` | 129, 133 | |
| `VEYSTRIX_FLASHNEXT_CONTEXT`, `_OUTPUT_TOKENS`, `_MTP_DEPTH` | 130–132 | |
| `VEYSTRIX_NEMOTRON_OUTPUT_TOKENS`, `_MTP_DEPTH`, `_MTP_PATH` | 134–136 | |
| `FLASHNEXT_AUTO_PURGE` | 141 | Hard-coded `1`; reaches Nemotron only (comment 137–140) |
| `VEYSTRIX_NEMOTRON_AUTHORIZED` | 143 | `1` only when Nemotron is selected — the real-inference approval |
| `VEYSTRIX_API_KEY`, `VEYSTRIX_SERVER_API_KEY` | 144–145 | |
| `PYTHONPATH` | 146 | Bundled `engine/lib` only |

**Safety-related guards specifically:** battery floor (98), Low Power Mode (99, plus Swift-side refusal
in `EngineController.swift:69–70` and `MemoryProbe.powerRefusal`), swap ceiling (108), invalid tool
calls (103), auto-purge (141), model handoff (107), Nemotron authorization (143), strict extended
context (116). A new "use ANE" toggle must not bypass any of these: in particular it should not set
`VEYSTRIX_NEMOTRON_AUTHORIZED` or relax the low-power/battery guards.

### Checkpoint / model selection

- `AppSettings.selectedModel` (`Settings.swift:123`), default `.flashnext`; two families in
  `ModelFamily` (34–48). Descriptors mirrored in `VeystrixKit/ModelIdentity.swift:16–30`
  (context limits 262,144 and 1,048,576; download repos are MLX 4-bit checkpoints).
- `activeCheckpointPath` (`Settings.swift:164–167`) picks `checkpointPath` vs `nemotronCheckpointPath`
  (kept independent).
- `AppModel.chooseCheckpoint` (`AppModel.swift:62–71`) auto-switches family via
  `CheckpointValidator.detectFamily` (`CheckpointValidator.swift:148`), called from `ModelView.swift:255`.
- `CheckpointValidator.validate` (`CheckpointValidator.swift:41+`) reads only small JSON metadata and
  `stat`s weight shards; it never opens weights (comment 3–4). Nemotron requires the MLX affine 4-bit
  conversion, not NVFP4 (68–75).
- `LaunchPlan.make` refuses a missing/incomplete checkpoint (48–51) and verifies bundled `.pyc.enc`
  resources (67–74).

### Where a "use ANE" setting would have to be threaded

1. `VeystrixKit/Settings.swift`: add a `Bool` field to `AppSettings` (near `activePerformance` /
   `flashNext*`), add its `CodingKeys` case (185–191), a `decodeIfPresent` line in `init(from:)` (193–232),
   and a cross-toggle rule in `validationProblems()` (235–254) if it must exclude existing speed profiles.
   Consider whether it belongs in `PerformanceOptions` (85–110), which is already per-model and fully
   emitted by the loop at `LaunchPlan.swift:122–128` — that is the lowest-friction route.
2. `Veystrix/SettingsView.swift`: add a Toggle in the Engine (15–46) or Performance (47–66) section.
3. `VeystrixKit/LaunchPlan.swift`: emit a new CLI arg (79–92) or env var (94–146). If placed as a new
   prefix loop entry it is emitted for both models; if ANE is Flash-Next-only, add a single
   `VEYSTRIX_FLASHNEXT_USE_ANE`-style variable near 130–132.
4. Engine side (out of scope of this app map): a consumer in `veystrix/model_router` / the model
   process for that arg/env.
5. Packaging: add any `.mlmodelc` to the bundled-resource existence check at `LaunchPlan.swift:67–74`.
6. Tests and docs: see §5 and §4/§7.

---

## 3. Build and packaging

### `packaging/engine/build_engine_bundle.sh` (124 lines)

- Output layout (3–7): `python/`, `lib/`, `licenses/`, `BUNDLE-MANIFEST.txt`.
- Apple-silicon-only build host (18); pinned CPython from `runtime.pin`, SHA-256 verified (21–27).
- Pinned deps from `pyproject.toml` (30–38); opencv removed (39–43), audio/SciPy stack removed
  (44–49), MLX headers/cmake removed (51), `__pycache__` removed (52–54), Tcl/Tk/IDLE/tests removed
  (55–57).
- Engine packages copied (59–66); exact import closure of bench utilities copied (67–75);
  `native_read.c` precompiled to `native_read.bundled.dylib` (76–82).
- **Compile then delete source** (84–90): `compileall -q -b` produces legacy `.pyc` beside sources, all
  `.py` and `.c` are deleted, and the build fails if any remain.
- **Encrypt** (92–96): `protect_engine.py` turns every `.pyc` into `.pyc.enc`; build fails if any
  plaintext `.pyc` remains.
- License texts (98–100), manifest of installed distributions (102–115), pip/setuptools/wheel removed
  (116–119), `smoke_import.py` (121–122). No model is loaded at any point (8–12).

### `packaging/engine/protect_engine.py` and `_protect.py`

- `protect_engine.py` (99 lines): random 32-byte key (52), copies `_protect.py` and writes
  `sitecustomize.py` (56–57), encrypts every `*.pyc` (59–63), compiles a shield dylib
  `_protect_key.dylib` from `shield.c` holding `base XOR sha256(clean_signals)` (65–77), self-tests
  derivation (80–86), removes caches and fails if plaintext bytecode remains (88–93).
- `_protect.py` (115 lines): the only readable code besides `sitecustomize.py`; an import hook
  (`_EncryptedFinder` 92–108) that decrypts `.pyc.enc` in memory with HMAC-SHA256 stream (41–66)
  and `marshal.loads` it (69–71). Explicitly "NOT unbreakable DRM" (6–7).

### `packaging/macos/build_app.sh` (95 lines)

- `swift build -c release --arch arm64` and binary copied/stripped (14–22); Sparkle framework (24–28).
- `LSMinimumSystemVersion` derived from the **MLX binary's** minos (30–37) — MLX 0.32.2 is macOS 26.2.
- Resources: licenses/notices/VERSION (45–52); engine `ditto`'d into `Contents/Resources/engine`
  and smoke-tested twice (54–67).
- **Integrity manifest** (68–92): generated BEFORE signing; prefixes `engine/lib/`, `Info.plist`,
  `VERSION`, `LICENSE.txt`, `THIRD_PARTY_NOTICES.md` (77); excludes `BundleIntegrity.json`,
  `BUILD-CHANNEL` (79) and `.dylib`/`.so` (80); writes `BundleIntegrity.json` (90). Swift mirror is
  `VeystrixKit/BundleIntegrity.swift:36–41,80–84`; app refuses to start on tamper at
  `AppModel.swift:26–29,119–120,131–132`.
- Final code signing via `sign_app.sh` (94).

### `packaging/macos/make_dmg.sh` (18 lines)

Creates a zip and UDZO DMG from `Veystrix.app` (9–16), includes `BUILD-INFO.txt` and legal text in the
DMG staging folder (13–14), and codesigns the DMG only for release signing (17).

### `scripts/verify_artifacts.py` (125 lines)

Post-build guardrails: app/engine dirs (31–36), `>=100` encrypted modules (label says 209) (37–38),
no plaintext `.pyc` / product `.py` / `.c` (39–44), only the two bootstrap files readable (45–47),
shield and native dylibs present (48–50), manifest pins `>100` files and recomputes it exactly like the
app (52–75), naive decompile fails (79–85), bundled engine loads (87–95), signature valid (97–98),
DMG exists and SHA-256 printed (100–106), and source-level check that the key is still bound to the
shield digest (108–110).

### What would need to change to ship a Core ML `.mlmodelc` alongside the engine

Build/packaging seams (no speculation on whether it would be faster):

1. `build_engine_bundle.sh`: add a conversion/copy step for the `.mlmodelc` before the compileall and
   encrypt steps (84–96), or copy it after encryption. Decide where it lives; if under `lib/`, it is
   automatically pinned by the integrity manifest and checked by `verify_artifacts.py`.
2. `packaging/engine/smoke_import.py:54–58`: add the `.mlmodelc` to the mandatory-resource list (or a
   new presence check).
3. `LaunchPlan.swift:67–74`: add the model file to the bundled-runtime existence check so a missing
   model fails at plan time rather than later.
4. `build_app.sh:30–37`: the minimum-OS derivation currently looks only at MLX `.so`; a Core ML model
   has its own deployment target that would need consideration. `sign_app.sh:26–32` signs nested
   Mach-O by `*.so`/`*.dylib`/`*.metallib`/executables; a `.mlmodelc` is a directory bundle with
   compiled model files and is not matched by that `find` (line 32), so nested signing / entitlement
   behaviour would need an explicit decision.
5. `BundleIntegrity` prefixes are path-based, so a `.mlmodelc` under `engine/lib/` is pinned
   automatically; if placed elsewhere (e.g. `Contents/Resources/Models`), both
   `BundleIntegrity.swift:36–41` and `build_app.sh:77` and `verify_artifacts.py:60` must be kept in
   sync, and large binary files should be weighed against manifest size.
6. `scripts/verify_artifacts.py`: add guardrails for the new artifact (presence, integrity, no stray
   source). Note its existing "no product source in bundle" check only looks at `.py`/`.c`, so a model
   file would not be flagged as source.
7. Legal/notices: `packaging/legal/generate_legal.py:241–245` and `legal/THIRD_PARTY_NOTICES.md:247–250`
   currently state "Model checkpoints are not included"; this text and the EULA would have to change.

---

## 4. Code protection and the "no source in DMG" policy

### `docs/security/ANTI_REVERSE_ENGINEERING.md` (80 lines)

- Openly states nothing shipped to a Mac is uncrackable (1–7).
- "What this build does" (9–20): stripped binary, integrity manifest (doc says 220 files), encrypted
  chat history, ad-hoc signature (explicitly not Developer ID).
- "What it does not do" (22–38): engine ships as encrypted `.pyc.enc`; key not stored, shield dylib
  holds `base XOR sha256(clean_signals)` recomputed at runtime (24–31); no `if detected { refuse }`
  branch; "No readable source ships; the private git repo keeps editable source off GitHub" (34).
- "Keeping source out of the download" (55–71): (1) private source repo, (2) "No source in the
  bundle" — compile to `.pyc`, precompile `native_read.c` to a dylib, delete all `.py`/`.c`, encrypt
  to `.pyc.enc`, fail if plaintext remains, tiny `_protect.py` + `sitecustomize.py` import hook;
  (3) behaviour is not source. Note "do not put secrets, licence checks or valuable business logic in
  the client" (69–71).
- Publishing (73–80): no token, no Developer ID, no notarization here; release workflow creates a
  draft release in the public repo for manual review.

### Policy on shipping model files / Core ML

There is **no dedicated anti-reverse-engineering rule that names Core ML or `.mlmodelc`**. However,
there is a consistent legal/design policy that model files are not shipped:

- `packaging/engine/build_engine_bundle.sh:10–12`: "no model weights"; "Excluded on purpose: …
  and every checkpoint."
- `docs/packaging/PACKAGING.md:13–16`: "every model checkpoint" excluded; "~113 GB checkpoint is
  never bundled or downloaded; the user chooses a folder."
- `legal/LICENSE_PROVENANCE.md:58–61`: "No weights are in the repository, the app or the release
  artifacts."
- `legal/FREEWARE-LICENSE.txt:32–33` ("Model Checkpoints are not part of the Software") and §7
  (108–116): the Software does not include model weights and does not download them without explicit
  action.
- `packaging/legal/generate_legal.py:241–245` / `legal/THIRD_PARTY_NOTICES.md:247–250`: "Model
  checkpoints are not included."
- ADRs constrain adding execution/model families: `docs/decisions/README.md` ADR-001 (exactly two
  families, 7–17), ADR-003 (single identity descriptors, 34–46), ADR-005 (no tokenizer swap this
  cycle, 61–75).

A `.mlmodelc` that embeds trained weights would function as a checkpoint and would therefore need an
explicit policy/legal decision; it would also no longer be true that "no model weights ship".

### "No source in DMG" in practice

- All engine `.py` files are compiled to legacy `.pyc` and deleted; all `.c` files are deleted;
  `build_engine_bundle.sh:90` asserts none remain. Then every `.pyc` is encrypted to `.pyc.enc`
  (`build_engine_bundle.sh:92–96`, `protect_engine.py:59–63`), and no plaintext `.pyc` may remain (96).
- The only readable Python files in the shipped `lib/` are the import-hook bootstrap
  `_protect.py` and `sitecustomize.py`, written by `protect_engine.py:56–57`; `verify_artifacts.py:41–47`
  asserts they are the only readable code. They contain no product logic (`_protect.py:3–7`).
- The Swift binary is stripped with `strip -x` (`build_app.sh:20–22`); docs note class/selector names
  remain (`ANTI_REVERSE_ENGINEERING.md:11–12`).
- The DMG contains the app (with `Contents/Resources/engine`), `LICENSE.txt`, `THIRD_PARTY_NOTICES.md`,
  and optionally `BUILD-INFO.txt` (`make_dmg.sh:11–16`). No editable source is in the archive.
- Caveat: `docs/packaging/PACKAGING.md:8–11` is stale — it still says `.pyc` are "compiled, not
  protected" and that `setting_*.py` stay as source. The current build script asserts no source of
  any kind remains and encrypts everything.

---

## 5. Swift tests and what they pin

- `apps/macos/Veystrix/Tests/VeystrixKitTests/KitTests.swift`: 1,069 lines, **96** Swift Testing
  `@Test` functions. Runs with no model or GPU (`Package.swift` comment).
- `apps/macos/Veystrix/Tests/WorkflowSmoke.swift` (65 lines): executable smoke of a mock workflow
  (start, pause, resume, report, cancel, history), not assertions about settings.

Count/structure pins most relevant to a new toggle:

| Test (line) | What it pins |
|---|---|
| `decodePoolOffersValidatedAndLargerTierOptions` (287) | `DecodePool.allCases.map(\.rawValue) == [0,1,2,3,4,6,8,12]` |
| `exactSpeedToggleIsIndependentAndPersists` (294) | default on, persisted, `--flashnext-exact-speed`, `VEYSTRIX_NEMOTRON_DENSE_GEMM=1` |
| `defaultsAreSafe` (316) | default port 8080, pool `.none`, confirmation, no validation problems |
| `roundTripAndOwnerOnlyPermissions` (321) | settings JSON round trip, 0600 |
| `oldAndPartialSettingsFilesStillLoad` (331) | missing keys fall back |
| `damagedSettingsAreKeptAndDefaultsUsed` (339) | damaged-file handling |
| `invalidPortAndQuotaAreReported` (346) | expects `validationProblems().count == 2` |
| `tieredCacheSelectionReachesTheRouter` (366) | `--flashnext-kv-cache tiered4`, `absolute-swap-only`, swap ceiling "6144" |
| `buildsLoopbackOnlyPlanWithoutStartingAnything` (389) | loopback, `--overnight`, no cloud keys, inherited `HOME`, `VEYSTRIX_PORT`, `FLASHNEXT_AUTO_PURGE` |
| `powerSettingsReachTheEngine` (402) | `VEYSTRIX_MIN_BATTERY_PERCENT`, `VEYSTRIX_ALLOW_LOW_POWER`; also pins `VEYSTRIX_MODEL_HANDOFF=auto`, `VEYSTRIX_INVALID_TOOL_CALLS=pass`, absence of `VEYSTRIX_REQUIRE_AC` (406–408) |
| `requireKeyPassesServerKeyOnlyWhenEnabled` (423) | API key never on command line |
| `bundledPlanRejectsNonExecutableAndMissingResources` (449) | list of required `.pyc.enc` paths |
| `qwenSpeedSelectionReachesTheRouter` (467), `qwenBoundedPrefill…` (478) | mutually exclusive toggles |
| `advisoriesFlagBatteryAndLowMemory` (745) | expects `MemoryProbe.advisories(...).count == 5` |
| `launchPlanSelectsNemotronAndAuthorizesIt` (812) / `flashNextPlanIsUnchanged` (824) | per-model args/env |
| `signedFilesAreExcludedFromTheManifest` (90), `modifiedFileIsDetected` (79), `addedAndRemovedFilesAreDetected` (103) | integrity manifest behaviour |

Tests that would need updating for a new toggle:
- `defaultsAreSafe` (316) and `oldAndPartialSettingsFilesStillLoad` (331) if the default is non-nil
  or a new key must be back-filled.
- `roundTripAndOwnerOnlyPermissions` (321) / `settingsSurviveAnUpdateBecauseTheyLiveInApplicationSupport`
  (350) if the new field is persisted.
- `exactSpeedToggleIsIndependentAndPersists` (294) if the new toggle interacts with speed/exact-speed.
- `buildsLoopbackOnlyPlanWithoutStartingAnything` (389) and the per-model launch tests (812/824) —
  add an assertion for the new arg/env; add a mutual-exclusion case if applicable.
- `bundledPlanRejectsNonExecutableAndMissingResources` (449) if a new `.mlmodelc` is added to the
  bundled-resource list.
- `invalidPortAndQuotaAreReported` (346) if `validationProblems()` gains a new rule.
- `KitTests.swift` has no test that asserts a total count of settings keys, so no hardcoded "number of
  settings" pin needs changing. `scripts/verify_artifacts.py:38,57` has count-style guardrails
  (encrypted modules `>=100`, pinned files `>100`) that are tolerant of additions.

---

## 6. Release state

- `docs/README.md:3` (updated 2026-11-16): two models (Qwen 3.8 Flash-Next default, Nemotron 3.5
  Lightning experimental); unit suites "on `main`": 988 shared, 2,289 Flash-Next, 203 Nemotron;
  Linux fast lane 455 no-MLX tests; Swift logic tests in CI.
- `docs/release/RELEASE_PROCESS.md:3–6`: **"no public release has been made."** Free app, no Apple
  Developer Program membership; Developer ID sections ready but not activated. Current release
  identity at line 42: version `1.0.0`, build `1000001`, tag `v1.0.0`, public name "Veystrix 1.0.0".
  Feed `updates/appcast.xml` does not exist yet (35–40); release workflow always creates a **draft**
  release (49–55).
- `docs/release/FREE_CHANNEL.md:3–4,8–19`: free channel is default (ad-hoc signed, not notarized,
  Gatekeeper warns once); SHA-256 in `SHA256SUMS.txt` is the integrity anchor.
- `CHANGELOG.md:3–24`: `Veystrix 1.0.0` is labelled "public release candidate", first public release,
  no model swap; protection/integrity work; 218 files pinned; version 1.0.0 / build 1000001.
  `CHANGELOG.md:26`: `Veystrix 3.8.1 (development, qwen3.8-flash-next)` — "Not released."
- Branch picture: checked out branch is `neural-engine`, 33 commits ahead of `main`, 0 behind (local
  commits include the protection work, exact-speed profile, larger memory tiers). `main` exists and
  is the baseline referenced by docs/README and CI; `.github/workflows/ci.yml:4` runs pushes on
  `main, qwen3.8-flash-next, nemotron3.5-lightning` — `neural-engine` is **not** in that list (PRs
  still trigger CI via `pull_request`). Thus `main` is the documented integration branch,
  `qwen3.8-flash-next` is the CHANGELOG-labelled development branch, and `neural-engine` is the
  active local topic branch for this work.
- Transition docs: there is no `docs/transition/` directory on `neural-engine` or `main`.
  `docs/decisions/README.md:19–25` (ADR-002) references `docs/transition/qwen-4-flash.md` on the
  **`qwen-4-flash`** branch (verified present there via `git ls-tree`); it is not in this checkout.
  The only locally named "transition" file is the unrelated Python test
  `veystrix/tests/test_effort_transitions.py`.

---

## 7. Repository-wide search: CoreML / MLModel / .mlmodelc / ANE / neural engine / MLX / Metal / GPU / device

Method: case-insensitive ripgrep/grep over the worktree, excluding `.git`, `.build`, `results`, `dist`,
`.venv`, `__pycache__`, and generated `out/`.

### What exists

- **ANE / Core ML — exactly one substantive source: `experiments/ane/REPORT.md`** (105 lines) plus
  six raw JSON result files in `experiments/ane/results/` (`e1_plan.json`, `e2_verify.json`,
  `e3_latency.json`, `e4_overlap.json`, `e4b_alone.json`, `e5.json`). These are committed and clean on
  `neural-engine`. The Python scripts the report references (`e1_placement.py`, `e2_verify.py`,
  `e3_latency.py`, `e4_overlap.py`, `e4b_alone.py`, `e5_*.py`, `verify_with_powermetrics.sh`) are **not**
  present in the current tree; the report itself says it was written on branch `development3.8` with
  experimental code "untracked and uncommitted" (lines 10–11).
  - Report status (3–7): **"evaluated and rejected (2026-11-07). ANE development is stopped."** The
    stated reasons: batch-1 predictors never reach the ANE; predictor is ~0.1% of a 1K prefill; ANE
    traffic slows bandwidth-bound GPU work by 3–17%; not bit-exact for routing; dense offload needs
    duplicate fp16 weights that do not fit. Stated limits: random weights (no accuracy claim), battery
    mode, short loops, Python `coremltools` call path, no hardware energy counter.
  - Verified placement (24–37): batch ≥16 predictors under `CPU_AND_NE` wake the ANE; batch-1 stays
    on CPU. Energy counters not obtained (37).
  - Latency/memory/overlap numbers are recorded in the report (41–76) — this note deliberately does
    not re-analyse them.
  - Recommendation (96–101): "Do not build an ANE expert-route predictor or ANE prefetch for
    Flash-Next"; "Do not use the ANE for any computation that feeds the model's output (routing, dense
    layers)"; keep as a closed research route; optional hardware power confirmation.
  - Called out in evidence summary: `docs/performance/overnight-2026-11-16.md:30` — "ANE / AMX
    prefill | Not attempted: fp16 ANE numerics would change outputs and the routed experts are streamed".
  - A binary PNG match (`docs/flashnext/graphics/Token trace - Xcode.png`) is incidental.
- **Core ML / coremltools**: appear only in `experiments/ane/REPORT.md` (file-count search: 1 file for
  each term). No runtime dependency in `pyproject.toml`, no Core ML import in `veystrix/` or
  `models/`, no `.mlmodel` / `.mlmodelc` file anywhere in the tree.
- **MLX**: pervasive (413 files) — it is the inference backend: pinned dependency, MLX 4-bit
  checkpoints in `ModelIdentity.swift:20,28` and `ModelView.swift:10,53–54,181–182`, MLX env flags in
  `LaunchPlan.swift:127` (`_MLX_AUTO_BUFFER_CAPS`) and `Settings.swift:91`, and docs under `docs/MLX/`
  (described in `docs/README.md:46` as "MLX Metal backend source notes; reference material, not
  workflow instructions").
- **Metal / GPU**: 186 / 183 files — MLX Metal kernels, `*_METAL_FMA_DEQUANT` (`LaunchPlan.swift:126`),
  `largerMetalBuffers` (`Settings.swift:91`, `SettingsView.swift:62`), `ModelView`/docs descriptions,
  `DecodeAdvisor.swift:53`, `docs/flashnext/gpu-gil-deadlock-and-watchdog.md`, and performance notes.
  `sign_app.sh:32` already signs nested `*.metallib` files.
- **device**: 82 files; in the Swift app it is overwhelmingly `FileHandle.nullDevice`
  (`EngineProcess.swift:37,60–62`, `HarnessView.swift:72,109`, `WorkflowClient.swift:191`) and
  Keychain `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` (`APIKeyStore.swift:44`,
  `ChatHistoryCipher.swift:77`), not a compute-device abstraction.
- No `MLComputePlan`, `NeuralEngine` framework import, `MLModel` class, `.mlmodelc` bundle, or ANE
  code exists in `apps/`, `veystrix/`, `models/`, `packaging/`, or `scripts/`.

### What does not exist

- No Core ML / ANE execution path in the app, engine, or packaging.
- No Core ML Swift/`coremltools` dependency in `pyproject.toml` or the bundled runtime.
- No `.mlmodel` / `.mlmodelc` artifact, conversion script, or signing/integrity handling for one.
- No "use ANE" setting, argument, or environment variable.
- No pre-existing test, doc, or legal text that anticipates shipping a Core ML model; the standing
  policy is the opposite (§4).

---

## Seams summary (realistic insertion points, no performance claims)

| Layer | Seam | Files / lines |
|---|---|---|
| Settings | new per-model or app-level toggle with validation | `Settings.swift:85–110,113–255`; migration via 193–232 |
| UI | one Toggle in Engine or Performance section | `SettingsView.swift:15–66` |
| Launch contract | new arg or env in the plan; bundled-artifact existence check | `LaunchPlan.swift:79–92,94–146,67–74` |
| Process | no change needed; env/args flow generically | `EngineProcess.swift:29–70` |
| Engine | consumer of the new flag (outside this map) | Python `veystrix.model_router` / model process |
| Packaging | add `.mlmodelc` copy, smoke check, signing decision, min-OS decision | `build_engine_bundle.sh:84–96`; `smoke_import.py:54–58`; `sign_app.sh:26–32`; `build_app.sh:30–37` |
| Integrity | auto-pinned if under `engine/lib/`, else update three mirrored prefix lists | `BundleIntegrity.swift:36–41`; `build_app.sh:77`; `verify_artifacts.py:60`; add guardrail in `verify_artifacts.py` |
| Tests | update defaults/round-trip/plan tests | see §5 |
| Legal/docs | model-checkpoint policy and notices must be revisited if weights ship | `FREEWARE-LICENSE.txt:32–33,108–116`; `LICENSE_PROVENANCE.md:58–61`; `generate_legal.py:241–245`; `docs/packaging/PACKAGING.md:13–16`; ADRs in `docs/decisions/README.md` |
| Research gate | prior documented decision to account for | `experiments/ane/REPORT.md:3–7,96–101` |

---

# Existing ANE experiment and release-state reconciliation

Follow-up research, 2026-11-17. Read in full: `experiments/ane/REPORT.md` (105 lines) and all six
result files `experiments/ane/results/{e1_plan,e2_verify,e3_latency,e4_overlap,e4b_alone,e5}.json`.
`experiments/ane/` contains **only** `REPORT.md` plus those six JSONs (47,693 bytes total). The
Python reproduction scripts named in the report (`e1_placement.py`, `e2_verify.py`, `e3_latency.py`,
`e4_overlap.py`, `e4b_alone.py`, `e5_numerics_and_shape.py`, `e5_memory.py`) and
`verify_with_powermetrics.sh` are **not present in this tree**; the report itself says the code was
"untracked and uncommitted" on branch `development3.8` (`REPORT.md:10`). No performance claims below
are mine — all numbers are quoted from the report and its JSONs.

## 1. What exactly was tested

- **Status/date:** `REPORT.md:3` — "**Status: evaluated and rejected (2026-11-07). ANE development is stopped.**"
  A correction the same day (line 16) fixed an earlier 40-layer extrapolation to 48 layers ("56 ms,
  270 MB" → "68 ms and 310 MB"); "The conclusion is unchanged."
- **Provenance:** line 10–11 — "Branch `development3.8`. Experimental code in `experiments/ane/`,
  **untracked and uncommitted**. No production file was changed, no model weights were downloaded or
  read, Q4 is untouched, `main` is untouched."
- **Hardware/OS:** line 18 — "M4 MacBook Air, 16 GB, macOS 27.2".
- **Toolchain:** line 18 — "Core ML via coremltools 9.0, MLX 0.32.2, in an isolated venv (the repo
  `.venv` is untouched)"; `e1_plan.json:6` confirms `"coremltools": "9.0"`. Reproduction recipe at
  line 105: `python3.12 -m venv V; V/bin/pip install coremltools mlx==0.32.2 numpy`.
- **Core ML API/call path:** line 21 — "Core ML is called through the Python `coremltools` path, so
  per-call figures include its marshalling. A native Swift or Objective-C caller may be cheaper. Not
  measured." Instrumentation: static `MLComputePlan` op plans; ANE driver `DevicePowerState` polled
  about 20 Hz in a parent process; `proc_pid_rusage` v6 `ri_neural_footprint`; IOReport "Energy Model"
  channels. `powermetrics` was not run (needs sudo).
- **Models:** small predictors built with **random weights**, as `.mlpackage` targets under the
  git-ignored `experiments/ane/out/` (e.g. `router_b1`, `router_b256`, `router_topk_b1`, `mlp_b1`,
  `matmul_b512_h2048`, `router3_*`, `matmul_1024x2560x2560`). Shapes: router `2560 → 512` experts
  with top-10 of 512 (as in Qwen3.8-Flash-Next), 3-layer look-ahead `2560 → 1536`, MLP
  `2560 → 1024 → 512`, and fp16 matmuls `1024×2560×2560` / `512×2048×2048`.
  Line 20: "Every predictor uses **random weights**. Latency, placement and memory do not depend on
  weight values. **Predictor accuracy was not measured**, and nothing below claims it." No real
  checkpoint was loaded.
- **Run conditions:** lines 19, 22 — "**On battery (75% to 68%), Low Power Mode off, no thermal
  warning.** These are preliminary numbers, not plugged-in numbers. A battery guard (stop below 25%)
  ran before every case." / "Short loops (1 to 4 s per case). Cases that compare devices ran in fresh
  processes. The overlap test ran 2 rounds in reversed order and the rounds agree to within 0.02 ms."
  *Minor internal inconsistency:* the JSON battery fields show 74% (`e1_plan.json:3`), 72% (e2),
  72→69% (e4), not 75→68%; the report's summary range is slightly wider than the raw data.
- Evidence tags (line 12): **[measured]** here, **[recorded]** from `docs/flashnext/research.md`,
  **[estimate]** arithmetic, not a result.

## 2. Measured results (exact numbers)

### ANE vs CPU vs GPU placement and execution
`REPORT.md:28–37` table, quoted:

| Signal | Result |
|---|---|
| `MLComputePlan` (static) | "at batch 1 every op is planned on **CPU** though the ANE is 'supported'; at batch 16 and up, linear and silu are planned on the ANE; `topk`'s cast op is CPU-only" |
| ANE driver `DevicePowerState`, ~20 Hz | "ANE-loaded models: up **100%** of the loop. CPU_ONLY, CPU_AND_GPU, MLX GPU and the batch-1 model under CPU_AND_NE: **0%**" |
| `proc_pid_rusage` v6 `ri_neural_footprint` | "2.7 to 18.4 MB for ANE-loaded models; **0** for CPU_ONLY, GPU, MLX, and batch-1 CPU_AND_NE" |

"**Verified:** batch >= 16 linear predictors loaded under `CPU_AND_NE` are placed on, and wake, the
Neural Engine. **Verified not on the ANE:** the batch-1 (decode-size) predicate, which silently stays
on the CPU under `CPU_AND_NE` (35 us, 0 MB neural footprint)." (line 35; the report says "predictor").
Raw corroboration: `e1_plan.json` — `router_b1` linear `preferred: MLCPUComputeDevice` with
`supported: [MLCPUComputeDevice, MLNeuralEngineComputeDevice]`; `router_topk_b1` cast supported by
CPU only; `router_b256` linear `preferred: MLNeuralEngineComputeDevice`. `e2_verify.json` —
`ane_up_fraction: 1.0` for "router b256 CPU_AND_NE" and "matmul 512x2048x2048 CPU_AND_NE", and
`neural_after_load: 2719744` / `10584064` bytes respectively; `0.0` / `0` for every CPU_ONLY,
CPU_AND_GPU, MLX and batch-1 CPU_AND_NE case. `e3_latency.json` `plan_preferred` is `"CPU"` at
batch 1 and `"NeuralEngine"` at batch ≥16.

### Latency (p50 ms per call) — report table at `REPORT.md:43–48`, verbatim

| Batch (tokens) | Core ML ANE | Core ML CPU | MLX GPU (bf16) | MLX CPU | numpy fp32 |
|---|---|---|---|---|---|
| 1 (decode) | 0.035 (**ran on CPU**) | 0.036 | 0.22 | 0.019 | 0.010 |
| 16 | 0.173 | 0.072 | 0.20 | 0.080 | 0.041 |
| 256 | **0.409** | 0.421 | 0.49 | 1.28 | 0.99 |
| 1024 | **1.41** | 1.61 | 1.37 | 6.47 | 4.98 |

Line 50: "MLX and numpy rows include a top-10 `argpartition`; the Core ML router rows do not, so only
ANE-vs-Core-ML-CPU is like for like." Raw p50s from `e3_latency.json`: `router_b1` coreml-cpu
`0.0355839729309082`, coreml-ane `0.035375007428228855`; `router_b16` `0.0718749943189323` /
`0.17320801271125674`; `router_b256` `0.4206669982522726` / `0.40912494296208024`; `router_b1024`
`1.6057919710874557` / `1.4114170335233212`. MLX GPU bf16 at 256/1024: `0.49279199447482824` /
`1.3696670066565275`; MLX CPU bf16 `1.2784579885192215` / `6.472375011071563`; numpy fp32
`0.9857079712674022` / `4.983042017556727`.

Larger predictors (line 52, verbatim): "look-ahead over 3 layers (2560 -> 1536) 0.61 vs 0.86 ms at
batch 256 and 2.22 vs 3.52 ms at 1024; MLP 2560 -> 1024 -> 512 0.50 vs 0.70 and 1.76 vs 2.90. So
the ANE is **1.15x to 1.6x faster than Core ML on the CPU only at batch >= 256 and with more than a
bare router**. A router with `topk` inside is **slower** on the ANE (1.09 vs 0.53 ms at 256) because
the plan splits CPU/ANE." Raw `router_topk_b256`: cpu `0.5275419680401683` vs ane
`1.0918329935520887` ms.

Fixed cost / pacing (line 54, verbatim): "ANE dispatch costs about 0.15 ms regardless of size (batch
16 is 0.173 ms for 42 MFLOP); Core ML on the CPU about 0.035 ms; an MLX GPU eval about 0.2 ms. A
paced ANE predictor (one call every 5 ms, about a layer's cadence) took **1.64 ms**, four times its
flat-out 0.41 ms, because the ANE clocks down between calls." Raw `e4b_alone.json` p50: 200 Hz
`1.6421250184066594` / `1.6429999959655106` ms; flat out `0.4072909941896796` /
`0.4070830182172358` ms. "Moving a live hidden state from MLX into Core ML also needs a GPU sync and
a copy; that was not measured on a live model."

### Throughput (fp16 matmuls, includes marshalling)
Line 56, verbatim: "1024x2560x2560 ANE **4.2 TFLOPS**, MLX GPU 3.1, Core ML CPU 2.5;
512x2048x2048 ANE 3.5, MLX GPU 2.8, CPU 2.2." Raw `e5.json`: `matmul_1024x2560x2560_coreml-cpu`
p50 `5.319375020917505` ms / `2.523186041070848` TFLOPS; `_coreml-ane` `3.1951250275596976` ms /
`4.200703473019016` TFLOPS / `18.448384` MB neural; `_mlx-gpu-fp16` `4.360000020824373` ms /
`3.0783882421776365` TFLOPS. `e2_verify.json` 512×2048×2048 p50s: CPU_ONLY `1.9568750285543501`,
CPU_AND_NE `1.2043750029988587`, CPU_AND_GPU `6.907041009981185`, MLX fp16 `1.5435420209541917` ms.

### Memory
Lines 60–62, verbatim: "Core ML runtime import: +59 MB RSS (31 -> 91 MB). **[measured]**" /
"One ANE router model: +25 MB RSS for the first, about +2 MB for each further one, **2.72 MB** ANE
footprint each (7.96 to 18.4 MB for larger models). The same model on CPU: +45 MB first, then about
+4 to 10 MB each. **[measured]**" / "**[estimate]** 48 per-layer routers on the ANE (the checkpoint
has 48 layers, all with routed experts): about 130 MB ANE footprint (48 x 2.72 MB) plus about 120 MB
RSS plus the 59 MB runtime, roughly **310 MB**. The recorded prefill left about 100 MB free and the
guard headroom is about 0.6 GB."
Raw `e5.json`: `memory_cpu` start RSS `31.522816` MB → after `import coremltools` `90.882048` MB →
`load+first predict #1` `136.249344` MB (neural 0); `memory_ane` start `31.45728` MB → import
`90.783744` MB → four successive loads RSS `115.9168 / 117.768192 / 119.504896 / 121.176064` MB
with neural `2.719744 / 5.439488 / 8.159232 / 10.878976` MB. `e3_latency.json` router b1024 ANE
`neural_MB: 7.962624`; `router_topk_b1024` `12.20608`; `mlp_b1024` `13.746176`.

### Energy
**Not measured.** Line 37, verbatim: "**Not obtained:** energy counters. IOReport 'Energy Model'
channels read zero for CPU, GPU and ANE even under a CPU burn, in the sandbox and (with your one-time
approval of that exact call) outside it. `powermetrics` needs sudo, so I did not run it;
`verify_with_powermetrics.sh` is ready if you want hardware power confirmation. The signals above show
the model is loaded on the ANE and the ANE is awake while it runs; they do not prove each individual
op executed there, and the `topk` model is a visible CPU/ANE split." No energy numbers exist in any
JSON.

### Overlap with GPU work
Report table at lines 68–73, verbatim (GPU worker's p50; mean of two reversed rounds; baselines
"GPU compute-bound (21.56 ms alone)" and "GPU bandwidth-bound, decode-like (8.13 ms alone)"):

| Side load | GPU compute-bound | GPU bandwidth-bound |
|---|---|---|
| ANE predictor, 200 Hz | +0.3% | **+3.5%** |
| ANE predictor, flat out | -0.2% | **+17.6%** |
| ANE saturating matmul | +0.5% | **+14.9%** |
| CPU Core ML predictor, flat out | +0.2% | +2.2% |

Line 75: "GPU and ANE **can** run at the same time, and compute-bound GPU work does not notice.
Bandwidth-bound GPU work (the shape of decode) **does** slow, and the ANE costs it more than a CPU
predictor does." Line 76: "The ANE slows under GPU load too: the saturating matmul took 1.22 ms alone
(two repeats) and 2.8 to 3.0 ms beside either GPU workload, 2.3x to 2.5x slower. The small predictor
moved less (0.407 ms alone, 0.425 to 0.45 beside the GPU)."

### Routing numerics
`e5.json`: `numerics_cpu` rows `2048`, `identical_top10_set: 0.8994140625`,
`mean_overlap: 0.989794921875`, `max_abs_logit_diff: 0.04901838302612305`,
`mean_abs_logit_diff: 0.003881655400618911`; `numerics_ane` rows `2048`,
`identical_top10_set: 0.98779296875`, `mean_overlap: 0.998779296875`,
`max_abs_logit_diff: 0.0021352767944335938`, `mean_abs_logit_diff: 0.0002393372415099293`.
Report line 80 rounds these to 98.8% (ANE) vs 89.9% (CPU fp16) identical top-10 sets over 2,048
random rows and adds: "not exact, so it would change routing. Random weights make the rate
indicative only."

## 3. Stated reason for rejection — verbatim

Status line 3–7: "**Status: evaluated and rejected (2026-11-07). ANE development is stopped.**
Reasons, all measured or recorded below: batch-1 predictors never reach the ANE; the predictor is not
the bottleneck (about 0.1% of a 1K prefill); ANE traffic slows bandwidth-bound GPU work by 3% to 17%;
the ANE is not bit-exact for routing; dense offload needs duplicate fp16 weights that do not fit.
Known limits: random weights (no accuracy claim), battery-mode preliminary runs, short loops, Python
Core ML call path, no hardware energy counter."

Section 5, lines 84–87 (why ANE prediction does not help): "(1) **Decode:** the predictor is not the
bottleneck and the ANE does not even run it (batch 1 stays on the CPU, 10 to 35 us) … a *perfect*
predictor with overlapped reads tied the baseline (+0.1%, -2.9%, -0.7%); early submit lost 8.2%; the
cause is memory-controller contention." "(2) **Prefill, 2048 tokens and up:** about 400 of 512
experts are routed per layer … so everything is staged and no prediction is needed." "(3) **Prefill,
1K:** … the 1024-token router costs 1.4 ms per layer, about 68 ms over 48 layers, 0.14% of the
recorded 49.3 s prefill. The cost is reads (25.6 s) and GPU drain (14.9 s), not prediction." "(4)
Cost of trying: about 310 MB **[estimate]**, a private framework dependency, and a GPU-to-ANE copy
per layer."

Section 6, lines 91–94: "**Dense prefill matmuls:** the ANE is 1.2x to 1.4x faster per FLOP
**[measured]**, but Core ML needs its own fp16 copy of the weights (the 3.4 GB dense core cannot be
duplicated here), fp16 differs from bf16 (not exact), and the experts are Q4 `gather_qmm`, which Core
ML cannot use. **[estimate]** even an ideal, exact, free-memory split of the 14.9 s GPU drain would
save at most about 5 s (11% of prefill, with the measured 2.4x concurrent ANE slowdown) to 9 s (17%,
ignoring that slowdown), and the real figure is smaller because most of that time is the MoE."
"**Routing itself:** not exact (above)." "**Draft model for speculation:** not tested. **[recorded]**
speculative decoding was break-even, and the ANE slows bandwidth-bound GPU work." "Nothing found that
is both useful and low in memory."

## 4. Permanently stopped, or just this attempt?

The report states the line as categorically as it could without using the word "permanently":
`REPORT.md:3` — "**Status: evaluated and rejected (2026-11-07). ANE development is stopped.**"
Recommendation 3 (line 100) is "Keep this as a closed route in the research log, alongside the
memory-controller contention finding." Recommendations 1 and 2 are absolute within their scope (lines
98–99): "**Do not build an ANE expert-route predictor or ANE prefetch for Flash-Next.**" and "**Do not
use the ANE for any computation that feeds the model's output** (routing, dense layers)".

The only opening is recommendation 4 (line 101), quoted verbatim: "**Optional, only if you want to
keep the question open:** run `sudo experiments/ane/verify_with_powermetrics.sh <python> ane` once for
hardware ANE power confirmation, and repeat the timing runs plugged in. I do not expect either to
change the recommendation."

So: as written, ANE development is **stopped and the route is treated as closed**; there is no stated
intention to resume, only an explicitly optional, not-expected-to-matter check. The report does not
literally say "never under any circumstances", and it explicitly notes (line 21) that a native
Swift/Objective-C caller was **not measured** — that is a scope limitation recorded by the author, not
a stated plan. Nothing in the report authorizes treating the decision as overturned.

## 5. MLComputeUnits cases and placement verification

The JSON `units` labels are `cpu`, `ane`, `gpu` (`e2_verify.json`) and `coreml-cpu`, `coreml-ane`
(`e3_latency.json`, `e5.json`); the `case` strings in `e2_verify.json` name the actual Core ML options:
**`CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`**. Those are three of the four standard `MLComputeUnits`
cases; **`ALL` (CPU + GPU + NE) was not used or reported anywhere** in the report or the six JSONs.
`REPORT.md:31` likewise enumerates only CPU_ONLY, CPU_AND_GPU and CPU_AND_NE.

Placement was verified with three independent non-root signals (report lines 26–37) plus timing:
`MLComputePlan` static op-level device plans (`e1_plan.json`, `plan_preferred` in `e3_latency.json`);
ANE driver `DevicePowerState` polled ~20 Hz (`ane_up_fraction` — `1.0` on the ANE-loaded cases,
`0.0` on CPU/GPU/batch-1); and `proc_pid_rusage` v6 `ri_neural_footprint` (`neural_MB`/`neural_after_*`).
**No**, it was not verified with hardware energy/power counters — those were explicitly "Not obtained"
(line 37); IOReport channels read zero and `powermetrics` was not run. The report also cautions that
these signals "show the model is loaded on the ANE and the ANE is awake while it runs; they do not
prove each individual op executed there".

## 6. In-repo follow-ups, TODOs, ADRs

- A repo-wide grep for the literal `experiments/ane` finds **only `REPORT.md` itself** (lines 10, 101,
  105). There is no follow-up note, TODO, issue stub, or ADR that cites the experiment.
- `docs/decisions/README.md` (ADR-001…ADR-005) contains **no ANE decision record**; ADR-001 instead
  says new model requests are evaluated as a swap of an existing family, and ADR-003 fixes the
  identity-descriptor process.
- The only other ANE mention under `docs/` is a one-line status in
  `docs/performance/overnight-2026-11-16.md:30`: "| ANE / AMX prefill | Not attempted: fp16 ANE
  numerics would change outputs and the routed experts are streamed |". This is consistent with the
  closed-route decision, not a follow-up task. (An incidental `Token trace - Xcode.png` binary also
  matches.) Sibling notes in `research_notes/Apple Neural Engine integration/` survey external
  literature on ANE/hybrid scheduling; none cites `experiments/ane`.
- The report's own disposition is the only "next step": keep it closed (line 100) plus the optional
  powermetrics check (line 101). Because the reproduction scripts and `verify_with_powermetrics.sh`
  are absent from the tree, `REPORT.md` + six JSONs are the entire durable artifact.

## 7. Release-state reconciliation (correcting my earlier note)

My earlier note in §6 said "no public release has been made" and characterized 1.0.0 as a "public
release candidate". That was an accurate report of the in-repo docs but **factually stale**. Both
sides of the contradiction need qualifying:

**What the local repo still says:**
- `docs/release/RELEASE_PROCESS.md:3–6` — "Status 2026-11-10: **no public release has been made.**"
- `docs/release/FREE_CHANNEL.md:41` — "Not yet tested on a downloaded copy of a published release (no
  release exists)."
- `CHANGELOG.md:3` — "Veystrix 1.0.0 (public release candidate)".
- `.github/workflows/release.yml:141,157` still creates a **DRAFT** release in the public repo for
  manual review; `veystrix/tests/test_distribution.py:60` asserts `--draft` is present.
- These documents are dated 2026-11-10/16 and were not updated after publication.

**What is actually true now (read-only live check, 2026-11-17):**
- `https://github.com/joyboyreturns/Veystrix/releases/tag/v1.0.0` resolves to a **published** release
  titled **"Veystrix 1.0.0 (experimental)"**, marked **Latest**, released **16 Nov 17:59** (© 2026),
  tag `v1.0.0`, commit `d55e3f5`. Body: "Experimental release. This is the first public build. Use it
  as a preview, not as a production tool." Assets: `Veystrix-1.0.0-arm64.dmg`,
  `Veystrix-1.0.0-arm64.zip`, `SHA256SUMS.txt`. It is the free-channel build (ad-hoc signed, **not**
  notarized, Gatekeeper warning on first open) and carries an "Update ( **repaired** )" note that the
  initial 1.0.0 build had an integrity-manifest bug ("3 files failed") and the DMG/ZIP were rebuilt
  on 2026-11-16.
- The release notes also say the signed Sparkle feed is **not** published yet — that part of
  `RELEASE_PROCESS.md:35–40` remains accurate.
- The two-repo split explains the apparent tag contradiction: `git remote -v` shows `origin` is the
  **private** `Veystrix-Core.git`. `git tag -l` lists only `pre-3.8-main-20261110` and
  `v3.8-baseline`; `git ls-remote --tags origin` returns the same two. The public repo's `v1.0.0`
  tag (`d55e3f5`) is **not** a local object and not on this private remote. The workflow publishes a
  draft to the public repo; the owner then published it by hand, which local docs never recorded.

**Verdict:** the "other records" are correct that a public v1.0.0 release exists; my earlier "no
public release has been made" statement was true only of the stale local docs. The accurate current
state is: **a published, experimental public v1.0.0 release exists (free channel, ad-hoc signed, not
notarized, with a repaired-asset note), while the in-repo docs still describe a pre-release state.**
The repaired integrity-manifest bug noted publicly is consistent with the local branch commit
`a70289f "Fix integrity manifest rejecting signed files at launch"`. No code files were changed
during this follow-up; the only modifications are the append to this notes file.

