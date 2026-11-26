# Contributing to Veystrix

Thank you for your interest in contributing to Veystrix.

All changes to `main` go through pull requests and require maintainer review
and passing CI checks before merging.

## Pull request workflow

1. Fork [Veystrix/veystrix-engine](https://github.com/Veystrix/veystrix-engine) and create a branch from `main`.
2. Keep changes focused and include unit tests for any new or modified behavior.
3. Run the local verification checks before opening a pull request:

```sh
./scripts/verify.sh --fast       # pure-Python unit tests + secret scan
```

On Apple Silicon macOS with MLX installed, you can also run the full test suite:

```sh
.venv/bin/python -m unittest discover -s veystrix/tests -t . -p 'test_*.py' -q
```

4. Open a pull request against `main` describing what changed and how you tested it.

## Repository guidelines

- Use plain, descriptive commit titles that start with a capital letter.
- Keep `main` green: CI runs pure-Python tests on Linux, full tests on macOS, and a secret scan on every push and pull request.
- Do not commit model weights, local cache artifacts, credentials, or personal paths.
- Preserve bit-exact greedy parity and memory/swap ceilings on 16 GB Apple Silicon hardware.

## License

By contributing to Veystrix, you agree that your contributions will be licensed
under the [Apache License 2.0](LICENSE).
