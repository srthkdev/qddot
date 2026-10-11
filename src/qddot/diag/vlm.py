"""Optional visual-language adapter; no provider is configured by default."""

from __future__ import annotations


def describe(images: list[str]) -> str:
    """Describe images through a configured provider when one is available."""
    del images
    raise RuntimeError("no visual-language provider is configured")
