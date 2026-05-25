"""
E2E UI-тесты (Selenium): регистрация, вход, выход, навигация.
"""
import uuid
import time

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

pytestmark = pytest.mark.selenium


def test_guest_sees_login_and_register_links(driver, base_url):
    driver.get(base_url + "/")
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.LINK_TEXT, "Войти")))
    assert driver.find_element(By.LINK_TEXT, "Регистрация").is_displayed()
    assert "Определитель AI-изображений" in driver.page_source


def test_register_and_login_flow(driver, base_url):
    suffix = uuid.uuid4().hex[:8]
    username = f"selenium_{suffix}"
    email = f"{username}@example.com"
    password = "testpass123"

    driver.get(base_url + "/register")
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.NAME, "username")))

    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "email").send_keys(email)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.NAME, "confirm").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    wait.until(EC.url_contains("/login"))
    assert "Вход" in driver.page_source

    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    time.sleep(2)
    wait.until(EC.url_contains("/dashboard"))
    assert "Панель управления" in driver.page_source


def test_logout_returns_to_index(driver, base_url):
    suffix = uuid.uuid4().hex[:8]
    username = f"logout_{suffix}"
    email = f"{username}@example.com"
    password = "testpass123"

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

    driver.get(base_url + "/logout")
    wait.until(EC.url_contains("/"))
    assert "Определитель AI-изображений" in driver.page_source


def test_wrong_password_shows_error(driver, base_url):
    driver.get(base_url + "/login")
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.NAME, "username")))
    driver.find_element(By.NAME, "username").send_keys("nobody_user_xyz")
    driver.find_element(By.NAME, "password").send_keys("wrong")
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "alert-danger")))
    assert "Неверное" in driver.page_source
