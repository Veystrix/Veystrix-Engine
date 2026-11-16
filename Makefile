.PHONY: verify fast python-only swift-test scan setup

verify:
	./scripts/verify.sh

fast:
	./scripts/verify.sh --fast

python-only:
	./scripts/verify.sh --python-only

swift-test:
	swift test --package-path apps/macos/Veystrix

scan:
	.venv/bin/python scripts/security/secret_scan.py

setup:
	./chat.sh setup
