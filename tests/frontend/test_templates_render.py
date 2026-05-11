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

