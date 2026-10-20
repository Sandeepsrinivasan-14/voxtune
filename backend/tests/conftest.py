import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import db  # noqa: E402
from app.core.gpu import GpuInfo  # noqa: E402
from app.core.config import settings  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # The API tests were written on a 4 GB RTX 3050; pin that so they behave the same on CPU-only CI.
    fake_gpu = lambda: GpuInfo(True, "RTX 3050", 4, 4)  # noqa: E731
    for mod in ("runs", "models", "dashboard"):
        monkeypatch.setattr(f"app.api.{mod}.detect_gpu", fake_gpu, raising=False)
    db.init_db(f"sqlite:///{tmp_path / 'test.db'}")
    from app.main import app

    with TestClient(app) as c:
        yield c
