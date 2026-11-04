"""Safe paths, filenames, and secret-safe errors."""

import re
from pathlib import Path


def safe_filename(name: str) -> str:
    value = Path(name.replace("\\", "/")).name
    value = re.sub(r"[^\w.() -]", "_", value).strip(" .")
    if not value or value in {".", ".."}:
        raise ValueError("Please provide a valid filename.")
    return value[:180]


def safe_error(error: BaseException, secrets: list[str | None] | None = None) -> str:
    message = str(error)
    for secret in secrets or []:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:1000]
