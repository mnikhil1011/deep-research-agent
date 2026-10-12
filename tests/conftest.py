"""Shared test setup."""

import pytest

from app.tools import scholar


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    """Fail any test that would reach a real scholarly service, and keep caches out of the repo."""

    def refuse(service, url, params=None):
        raise AssertionError(f"a test tried to call {service}: {url}")

    monkeypatch.setattr(scholar, "_get", refuse)
    monkeypatch.setattr(scholar.config, "CACHE_DIR", str(tmp_path / "cache"))
