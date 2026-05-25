# tests/e2e/test_navigation_ui.py
import uuid
import time
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

pytestmark = pytest.mark.selenium

def _register_and_login(driver, base_url, username, password):
    email = f"{username}@example.com"
    driver.get(base_url + "/register")
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.NAME, "username")))
    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "email").send_keys(email)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.NAME, "confirm").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    time.sleep(2)
    wait.until(EC.url_contains("/login"))
    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    wait.until(EC.url_contains("/dashboard"))

def test_navigate_history_and_profile(driver, base_url):
    username = f"nav_{uuid.uuid4().hex[:8]}"
    _register_and_login(driver, base_url, username, "testpass123")
    wait = WebDriverWait(driver, 10)

    # Проверка страницы истории
    driver.get(base_url + "/history")
    # Ищем заголовок h1 или любой текст вместо h2
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
    assert "История" in driver.page_source or "проверок" in driver.page_source

    # Проверка страницы профиля
    driver.get(base_url + "/profile")
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h3")))
    assert "Мой профиль" in driver.page_source or "профиль" in driver.page_source.lower()