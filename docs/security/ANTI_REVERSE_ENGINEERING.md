# Reverse engineering and tamper resistance (honest limits)

Reviewed 2026-11-16. **Nothing distributed to a user's Mac is uncrackable.**
Tools like [REA](https://github.com/morluto/rea), Hopper, Ghidra, otool and
simple `unzip` can inspect any local app. The goal is to raise the cost of
casual copying, detect tampering, and protect the *business* logic that cannot
be extracted — not to make analysis impossible.

## What this build does

- **Stripped release binary** (`strip -x` in `build_app.sh`): removes local
  symbol names; class/selector names in Swift and the Objective-C runtime stay.
- **Integrity manifest** (`VeystrixKit/BundleIntegrity.swift`): the app pins
  SHA-256 digests of 220 bundled code/resource files and refuses to start the
  engine if any are changed, added or removed. Manifest is generated at build
  time before signing and only covers signed-stable files.
- **Encrypted chat history** (`ChatHistoryCipher.swift`): AES-256-GCM with a
  Keychain-backed key, so prompts are not readable as text on disk.
- **Ad-hoc code signature**: proves the bytes have not changed since build;
  it is **not** a Developer ID signature and does not stop modification.

## What it does not do

- The engine ships as **encrypted bytecode** (`.pyc.enc`). Each build generates
  a random 256-bit key compiled into a small native dylib; a minimal import
  hook decrypts modules in memory at load time. A decompiler now fails with
  `bad marshal data` instead of recovering docstrings, strings and logic.
  No readable source ships: settings providers load from encrypted bytecode,
  and `native_read.c` is compiled to a bundled dylib. The private git repo is
  what keeps the editable source off GitHub.
- A user with local access can call the same key dylib, patch the import hook,
  or remove the integrity check. Encryption stops decompilers, not a debugger.
  It raises the cost from "unzip and disassemble" to "reverse the bootstrap".
- No license keys, activation servers or entitlement checks are implemented.

## Real protection for a commercial product

1. **Developer ID + notarization**: add Apple credentials and a Sparkle EdDSA
   key, then run `make_release.sh --notarize`. Gatekeeper then checks the
   signature at launch. Still not uncrackable, but the trusted path.
2. **Server-side entitlement**: keep the valuable logic (licensing, model
   routing decisions, usage accounting) behind an authenticated service the
   app calls. A cracked client cannot forge a server-side decision.
3. **Licensing**: implement signed license files or an activation service with
   offline grace periods, plus audit logging. Enforce commercially, not by
   relying on binary secrecy.
4. **Legal**: keep MIT notices for upstream code (see
   `legal/COMMERCIALIZATION_READINESS.md`); use a license that forbids
   redistribution and reverse engineering where enforceable.

## Keeping source out of the download

Three layers, in order of importance:

1. **Private source repo.** Keep `Veystrix-Core` private; publish only release
   artifacts. This is the only thing that actually hides editable source.
2. **No source in the bundle.** `build_engine_bundle.sh` compiles every module
   to legacy `.pyc`, precompiles `native_read.c` into
   `native_read.bundled.dylib`, deletes all `.py` and `.c` files, then encrypts
   every `.pyc` into `.pyc.enc` with a per-build key compiled into
   `_protect_key.dylib`. The build fails if any plaintext bytecode or source
   remains. A tiny import hook (`_protect.py` + `sitecustomize.py`) decrypts in
   memory; source checkouts never use it. This also removes the old runtime
   `xcrun clang` dependency.
3. **Behaviour is not source.** A determined user can still observe what the
   app does, dump strings from the binary and decompile bytecode. Do not put
   secrets, licence checks or valuable business logic in the client.

## Publishing

The public `Veystrix` repo is separate from private `Veystrix-Core`. This
environment has no GitHub token, no Developer ID identity and no notarization
credentials, so the DMG here is a free-channel build and has not been uploaded.
The release workflow tags in `Veystrix-Core` create a **draft** release in the
public repo for manual review — use that path with the proper secrets rather
than attaching binaries by hand.
