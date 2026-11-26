# Starting Veystrix manually (safe commands)

Never launch the engine unsupervised; an unsupervised direct launch once exhausted memory and froze the Mac.
AC power is recommended for long sessions, but the engine also starts on battery: it refuses to start (and the
supervisor will not restart it) when the battery is below 25% (`VEYSTRIX_MIN_BATTERY_PERCENT`, `0` removes the floor), and the
in-run watchdog still stops a run whose battery falls below the floor. Switch Low Power Mode off and close heavy apps.
If memory is short the engine purges the macOS file cache by itself (conditional, never during a request; uses the
narrow passwordless `sudo -n /usr/sbin/purge` rule; `--no-auto-purge` turns it off).

```sh
cd ~/Documents/Veystrix
./serve.sh --status                                   # read-only: what the supervisor did last
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2         # one supervised run; Ctrl-C stops it
VEYSTRIX_API_KEY=local ./serve.sh --overnight         # supervised, restarts after a guard stop (bounded)
curl -s http://127.0.0.1:8080/health ; curl -s http://127.0.0.1:8080/v1/models
```

The server listens on `127.0.0.1` only. Checkpoint default: the owner's configured `qwen4_exp` folder; override with
`--checkpoint DIR`. These commands load the real model: **do not run them while the owner is away.**

The macOS app was removed on 2026-11-18 (see `docs/product/APP_FEATURE_INVENTORY.md`). Start the
engine from the `veystrix` terminal tool with `veystrix start --model …` after choosing a checkpoint;
the engine itself is never started implicitly.
