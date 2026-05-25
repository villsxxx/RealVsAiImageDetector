"""
Фикстуры Selenium: браузер и тестовый HTTP-сервер.

По умолчанию поднимается mini_app (без torch). Чтобы тестировать полный app.py:
  set SELENIUM_BASE_URL=http://127.0.0.1:5000
  запустите вручную: py Interface/app.py
  py -m pytest tests/selenium -m selenium
"""
# tests/e2e/conftest.py
import os
import sys
import socket
import threading
import time
from pathlib import Path

# ДОБАВИТЬ ЭТИ СТРОКИ В САМОМ НАЧАЛЕ (до всех импортов)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Теперь остальные импорты
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from werkzeug.serving import make_server

# Этот импорт должен работать
from mini_app import create_app


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server_url(tmp_path_factory):
    """URL тестового сервера: внешний (env) или локальный mini_app."""
    external = os.environ.get("SELENIUM_BASE_URL", "").strip()
    if external:
        yield external.rstrip("/")
        return

    db_dir = tmp_path_factory.mktemp("selenium_db")
    db_file = db_dir / "test.db"
    app = create_app(db_path=str(db_file))
    port = _free_port()
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    time.sleep(0.3)
    yield base
    server.shutdown()


@pytest.fixture
def base_url(live_server_url):
    return live_server_url


@pytest.fixture
def driver():
    headless = os.environ.get("SELENIUM_HEADLESS", "1") != "0"
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")

    browser = webdriver.Chrome(options=opts)
    browser.implicitly_wait(5)
    yield browser
    browser.quit()
