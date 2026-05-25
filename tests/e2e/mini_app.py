"""
Минимальное Flask-приложение для Selenium E2E.

Повторяет маршруты авторизации и основные страницы без импорта predictor/torch.
Основной Interface/app.py не используется.
"""
import uuid
import os
import sys
from pathlib import Path

from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

_INTERFACE = Path(__file__).resolve().parents[2] / "Interface"
if str(_INTERFACE) not in sys.path:
    sys.path.insert(0, str(_INTERFACE))

from models import db, User, Prediction  # noqa: E402
from forms import LoginForm, RegisterForm, ProfileCutoffForm  # noqa: E402


def create_app(db_path=None):
    app = Flask(
        __name__,
        template_folder=str(_INTERFACE / "templates"),
        static_folder=str(_INTERFACE / "static"),
    )
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "selenium_test.db")

    app.config.update(
        SECRET_KEY="selenium-test-secret",
        TESTING=False,
        WTF_CSRF_ENABLED=False,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    db.init_app(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "login"

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = RegisterForm()
        if form.validate_on_submit():
            hashed = generate_password_hash(form.password.data)
            user = User(
                username=form.username.data,
                email=form.email.data,
                password_hash=hashed,
            )
            db.session.add(user)
            db.session.commit()
            flash("Регистрация прошла успешно!", "success")
            return redirect(url_for("login"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and check_password_hash(user.password_hash, form.password.data):
                login_user(user)
                flash("Вы успешно вошли!", "success")
                return redirect(url_for("dashboard"))
            flash("Неверное имя пользователя или пароль", "danger")
        return render_template("login.html", form=form)

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("Вы вышли из системы.", "info")
        return redirect(url_for("index"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        recent = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.timestamp.desc())
            .limit(6)
            .all()
        )
        stats_total = Prediction.query.filter_by(user_id=current_user.id).count()
        stats_real = Prediction.query.filter_by(user_id=current_user.id, class_predicted=0).count()
        stats_ai = Prediction.query.filter_by(user_id=current_user.id, class_predicted=1).count()
        first_pred = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.timestamp.asc())
            .first()
        )
        days_activity = ((datetime.utcnow() - first_pred.timestamp).days + 1) if first_pred else 0
        return render_template(
            "dashboard.html",
            recent_predictions=recent,
            stats_total=stats_total,
            stats_real=stats_real,
            stats_ai=stats_ai,
            days_activity=days_activity,
        )

    @app.route("/history")
    @login_required
    def history():
        page = request.args.get("page", 1, type=int)
        pagination = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.timestamp.desc())
            .paginate(page=page, per_page=20)
        )
        return render_template("history.html", pagination=pagination)


    @app.route("/clear_history", methods=["POST"])
    @login_required
    def clear_history():
        """Очистить историю проверок пользователя"""
        Prediction.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash("История проверок очищена", "success")
        return redirect(url_for("history"))

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        form = ProfileCutoffForm(obj=current_user)
        if form.validate_on_submit():
            current_user.uncertainty_cutoff_percent = form.uncertainty_cutoff_percent.data
            db.session.commit()
            flash("Порог отсечения сохранён.", "success")
            return redirect(url_for("profile"))
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        predictions_count = len(predictions)
        real_count = sum(1 for p in predictions if p.class_predicted == 0)
        ai_count = sum(1 for p in predictions if p.class_predicted == 1)
        avg_confidence = (
            sum(p.confidence for p in predictions) / predictions_count
            if predictions_count
            else 0
        )
        return render_template(
            "profile.html",
            predictions_count=predictions_count,
            real_count=real_count,
            ai_count=ai_count,
            avg_confidence=avg_confidence,
            cutoff_form=form,
        )

    with app.app_context():
        db.create_all()

    return app
