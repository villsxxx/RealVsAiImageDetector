from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    batch_predictions = db.relationship('BatchPrediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(300), nullable=False)
    class_predicted = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class BatchPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    batch_type = db.Column(db.String(50), nullable=False)  # 'multiple', 'video', 'url_batch'
    source_filename = db.Column(db.String(300))  # Исходное имя файла для видео
    total_items = db.Column(db.Integer, default=0)
    summary_stats = db.Column(db.JSON)  # Храним статистику в JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    frames = db.relationship('FramePrediction', backref='batch', lazy=True, cascade='all, delete-orphan')

class FramePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch_prediction.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(300), nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    class_predicted = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)