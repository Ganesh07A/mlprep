"""Dataset validation.

Raises ``ValueError`` on invalid inputs — no UI or CLI concerns here.
The CLI boundary is responsible for turning errors into user-facing messages.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv"})


def validate_input(file_path: Path) -> None:
    """Validate that *file_path* is a supported, readable file.

    Args:
        file_path: Path to validate.

    Raises:
        ValueError: If the file does not exist or has an unsupported extension.
    """
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported formats: {supported}"
        )
