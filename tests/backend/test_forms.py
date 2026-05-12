import sys

import pytest
from wtforms.validators import ValidationError


class _Field:
    def __init__(self, data):
        self.data = data


def test_register_form_duplicate_username_validator(flask_app):
    """
    Проверяем именно наш кастомный валидатор, не полагаясь на email_validator,
    который может отсутствовать в окружении.
    """
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")  # .../Interface/
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models
    import forms

    with flask_app.app_context():
        existing = models.User(username="taken", email="x@example.com", password_hash="x")
        models.db.session.add(existing)
        models.db.session.commit()

        form = forms.RegisterForm()
        with pytest.raises(ValidationError):
            form.validate_username(_Field("taken"))


def test_register_form_new_username_ok(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")  # .../Interface/
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms

    with flask_app.app_context():
        form = forms.RegisterForm()
        # не должно выбрасывать исключение
        form.validate_username(_Field("free_username"))


def test_register_form_duplicate_email_validator(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models
    import forms

    with flask_app.app_context():
        existing = models.User(username="user1", email="busy@example.com", password_hash="x")
        models.db.session.add(existing)
        models.db.session.commit()

        form = forms.RegisterForm()
        with pytest.raises(ValidationError):
            form.validate_email(_Field("busy@example.com"))


def test_register_form_new_email_ok(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms

    with flask_app.app_context():
        form = forms.RegisterForm()
        form.validate_email(_Field("free@example.com"))


def test_login_form_requires_username_and_password(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms

    with flask_app.test_request_context("/login", method="POST"):
        form = forms.LoginForm(data={"username": "", "password": ""})
        assert form.validate() is False


def test_profile_cutoff_form_accepts_edges_and_rejects_out_of_range(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import forms

    with flask_app.test_request_context("/profile", method="POST"):
        assert forms.ProfileCutoffForm(data={"uncertainty_cutoff_percent": 50}).validate() is True
        assert forms.ProfileCutoffForm(data={"uncertainty_cutoff_percent": 100}).validate() is True
        assert forms.ProfileCutoffForm(data={"uncertainty_cutoff_percent": 49}).validate() is False
        assert forms.ProfileCutoffForm(data={"uncertainty_cutoff_percent": 101}).validate() is False

