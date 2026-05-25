# tests/e2e/test_profile_ui.py
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


def test_profile_update_cutoff(driver, base_url):
    """Обновление порога отсечения в профиле"""
    username = f"profile_{uuid.uuid4().hex[:8]}"
    _register_and_login(driver, base_url, username, "testpass123")

    driver.get(base_url + "/profile")
    wait = WebDriverWait(driver, 10)

    # Найти поле порога отсечения
    cutoff_input = wait.until(EC.presence_of_element_located((By.NAME, "uncertainty_cutoff_percent")))
    cutoff_input.clear()
    cutoff_input.send_keys("75")

    # Найти кнопку сохранения
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Сохранить']")

    # ПРОСКРОЛЛИТЬ до кнопки
    driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
    time.sleep(0.5)  # небольшая пауза после скролла

    # Кликнуть через JavaScript (обходит перехват клика)
    driver.execute_script("arguments[0].click();", submit_button)

    # Ждем сообщение об успехе
    time.sleep(2)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "alert-success")))
    assert "Порог отсечения сохранён" in driver.page_source

def test_profile_statistics_display(driver, base_url):
    """Проверка отображения статистики в профиле"""
    username = f"stats_{uuid.uuid4().hex[:8]}"
    _register_and_login(driver, base_url, username, "testpass123")

    driver.get(base_url + "/profile")
    wait = WebDriverWait(driver, 10)

    # Проверить, что статистика отображается
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h3")))
    assert "Всего проверок" in driver.page_source or "статистика" in driver.page_source.lower()