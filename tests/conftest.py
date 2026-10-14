import pytest

from building_with_rag import main
from building_with_rag.settings import Settings


@pytest.fixture(autouse=True)
def open_chat_by_default(monkeypatch):
    """Tests never depend on a developer's .env: chat auth is off unless a test turns it on."""
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None))
