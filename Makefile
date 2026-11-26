.PHONY: verify fast python-only artifacts scan bundle-report setup

verify:
	./scripts/verify.sh

fast:
	./scripts/verify.sh --fast

python-only:
	./scripts/verify.sh --python-only

artifacts:
	./scripts/verify.sh --artifacts

scan:
	.venv/bin/python scripts/security/secret_scan.py

bundle-report:
	.venv/bin/python scripts/packaging/bundle_size_report.py

setup:
	./chat.sh setup
