import sqlite3
import hashlib
from datetime import datetime

DB_NAME = 'detector.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS requests 
                 (id INTEGER PRIMARY KEY, user_id INTEGER, image_path TEXT, 
                  status TEXT, result TEXT, confidence REAL, 
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  ('admin', hashlib.sha256('admin'.encode()).hexdigest()))
    except:
        pass
    conn.commit()
    conn.close()

def get_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hashlib.sha256(password.encode()).hexdigest()))
    user = c.fetchone()
    conn.close()
    return user

def add_request(user_id, image_path):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO requests (user_id, image_path, status) VALUES (?, ?, 'pending')",
              (user_id, image_path))
    req_id = c.lastrowid
    conn.commit()
    conn.close()
    return req_id

def get_requests(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    reqs = c.fetchall()
    conn.close()
    return reqs

def update_request(req_id, status, result=None, confidence=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE requests SET status=?, result=?, confidence=? WHERE id=?",
              (status, result, confidence, req_id))
    conn.commit()
    conn.close()