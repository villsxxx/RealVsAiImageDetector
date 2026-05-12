def _register_dummy_routes(app):
    @app.route("/", endpoint="index")
    def _index():
        return "ok"

    @app.route("/login", endpoint="login")
    def _login():
        return "ok"

    @app.route("/register", endpoint="register")
    def _register():
        return "ok"

    @app.route("/dashboard", endpoint="dashboard")
    def _dashboard():
        return "ok"

    @app.route("/history", endpoint="history")
    def _history():
        return "ok"

    @app.route("/profile", endpoint="profile")
    def _profile():
        return "ok"

    @app.route("/logout", endpoint="logout")
    def _logout():
        return "ok"

    @app.route("/upload", endpoint="upload", methods=["POST"])
    def _upload():
        return "ok"

    @app.route("/clear_history", methods=["POST"], endpoint="clear_history")
    def _clear_history():
        return "ok"

    @app.route("/delete_prediction/<int:pred_id>", methods=["POST"], endpoint="delete_prediction")
    def _delete_prediction(pred_id):
        return "ok"

    @app.route("/prediction/<int:pred_id>", endpoint="prediction_detail")
    def _prediction_detail(pred_id):
        return "ok"


class _FakePaginationEmpty:
    """Минимальная заглушка пагинации для пустой истории."""

    page = 1
    items = []
    has_prev = False
    has_next = False
    prev_num = None
    next_num = None

    def iter_pages(self, left_edge=1, right_edge=1, left_current=2, right_current=2):
        yield 1


def test_templates_render_without_auth(flask_app):
    from flask import render_template

    _register_dummy_routes(flask_app)

    with flask_app.test_request_context("/"):
        html = render_template("index.html")
        assert "Определитель AI-изображений" in html


def test_login_and_register_templates_render_with_forms(flask_app):
    from flask import render_template

    _register_dummy_routes(flask_app)

    import sys

    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms

    with flask_app.test_request_context("/login"):
        html = render_template("login.html", form=forms.LoginForm())
        assert "Вход" in html

    with flask_app.test_request_context("/register"):
        html = render_template("register.html", form=forms.RegisterForm())
        assert "Регистрация" in html


def test_dashboard_template_renders_with_empty_stats(flask_app):
    from flask import render_template

    _register_dummy_routes(flask_app)

    with flask_app.test_request_context("/dashboard"):
        html = render_template(
            "dashboard.html",
            recent_predictions=[],
            stats_total=0,
            stats_real=0,
            stats_ai=0,
            days_activity=0,
        )
        assert "Панель управления" in html


def test_history_template_renders_empty_table(flask_app):
    from flask import render_template

    _register_dummy_routes(flask_app)

    with flask_app.test_request_context("/history"):
        html = render_template("history.html", pagination=_FakePaginationEmpty())
        assert "История проверок" in html


def test_profile_template_renders_for_logged_in_user(flask_app):
    import sys
    from datetime import datetime

    from flask import render_template
    from flask_login import login_user

    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms
    import models

    _register_dummy_routes(flask_app)

    with flask_app.app_context():
        models.db.session.add(
            models.User(
                username="profile_user",
                email="profile_user@example.com",
                password_hash="x",
                created_at=datetime.utcnow(),
            )
        )
        models.db.session.commit()

    with flask_app.test_request_context("/profile"):
        u = models.User.query.filter_by(username="profile_user").first()
        assert u is not None
        login_user(u)
        form = forms.ProfileCutoffForm(obj=u)
        html = render_template(
            "profile.html",
            predictions_count=0,
            real_count=0,
            ai_count=0,
            avg_confidence=0,
            cutoff_form=form,
        )
        assert "Мой профиль" in html


def test_result_template_renders_for_logged_in_user(flask_app):
    import sys
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from flask import render_template
    from flask_login import login_user

    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models

    _register_dummy_routes(flask_app)

    with flask_app.app_context():
        models.db.session.add(
            models.User(
                username="result_user",
                email="result_user@example.com",
                password_hash="x",
                created_at=datetime.utcnow(),
            )
        )
        models.db.session.commit()

    pred = SimpleNamespace(
        id=1,
        image_filename="sample.jpg",
        confidence=0.82,
        class_predicted=1,
        timestamp=datetime.now(timezone.utc),
    )

    with flask_app.test_request_context("/prediction/1"):
        u = models.User.query.filter_by(username="result_user").first()
        assert u is not None
        login_user(u)
        html = render_template("result.html", pred=pred)
        assert "Результат анализа" in html

