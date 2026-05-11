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

