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

- The engine ships as `.pyc`. Python bytecode decompiles easily; `.pyc` is
  anti-clutter, **not obfuscation**. Five settings modules and `native_read.c`
  remain as source because the loader reads them that way.
- A user with local access can remove the integrity check, patch the binary or
  swap the manifest. The check catches casual changes and corrupted downloads.
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

## Publishing

The public `Veystrix` repo is separate from private `Veystrix-Core`. This
environment has no GitHub token, no Developer ID identity and no notarization
credentials, so the DMG here is a free-channel build and has not been uploaded.
The release workflow tags in `Veystrix-Core` create a **draft** release in the
public repo for manual review — use that path with the proper secrets rather
than attaching binaries by hand.
