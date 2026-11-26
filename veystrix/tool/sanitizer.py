"""Conservative redactor for diagnostics and leak scans (port of Sanitizer.swift).

When in doubt, redact. Order matters: specific token shapes first, then generic
key=value forms, then prompt text, then home paths.
"""
from __future__ import annotations

import os
import re

PATTERNS = [
    (r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [REDACTED]"),
    (r"\bsk-ant-[A-Za-z0-9_-]{16,}", "[REDACTED-KEY]"),
    (r"\bsk-[A-Za-z0-9_-]{16,}", "[REDACTED-KEY]"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED-KEY]"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}", "[REDACTED-KEY]"),
    (r"\bhf_[A-Za-z0-9]{20,}", "[REDACTED-KEY]"),
    (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED-KEY]"),
    (r"\bAIza[0-9A-Za-z_-]{30,}", "[REDACTED-KEY]"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}", "[REDACTED-KEY]"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}", "[REDACTED-JWT]"),
    (r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*)\s*[=:]\s*[\"']?[^\s\"',;]+", r"\1=[REDACTED]"),
    (r"(?i)\"(api[_-]?key|authorization|token|secret|password)\"\s*:\s*\"[^\"]*\"", r'"\1":"[REDACTED]"'),
    (r"(?i)\"(content|prompt|text|input)\"\s*:\s*\"(?:[^\"\\]|\\.){0,20000}\"", r'"\1":"[PROMPT TEXT REMOVED]"'),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED-PRIVATE-KEY]"),
]
COMPILED = [(re.compile(pattern), replacement) for pattern, replacement in PATTERNS]


def clean(text: str, home: str | None = None, extra_secrets=()) -> str:
    result = text
    for secret in extra_secrets:
        if len(secret) >= 4:
            result = result.replace(secret, "[REDACTED]")
    for regex, replacement in COMPILED:
        result = regex.sub(replacement, result)
    home = home or os.path.expanduser("~")
    if home and home != "/":
        result = result.replace(home, "~")
    result = re.sub(r"/Users/[^/\s\"']+", "/Users/[user]", result)
    return result
