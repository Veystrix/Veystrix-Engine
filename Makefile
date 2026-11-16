.PHONY: verify fast python-only artifacts swift-test scan setup results-index bundle-report

verify:
	./scripts/verify.sh

fast:
	./scripts/verify.sh --fast

python-only:
	./scripts/verify.sh --python-only

artifacts:
	./scripts/verify.sh --artifacts

swift-test:
	swift test --package-path apps/macos/Veystrix

scan:
	.venv/bin/python scripts/security/secret_scan.py

results-index:
	.venv/bin/python scripts/results/index_citations.py --write

bundle-report:
	.venv/bin/python scripts/packaging/bundle_size_report.py

setup:
	./chat.sh setup
