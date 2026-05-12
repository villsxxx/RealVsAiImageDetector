import sys

import pytest
from sqlalchemy.exc import IntegrityError


def test_user_and_prediction_models(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")  # .../Interface/
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models

    with flask_app.app_context():
        u = models.User(username="u1", email="u1@example.com", password_hash="x")
        models.db.session.add(u)
        models.db.session.commit()

        assert u.id is not None
        assert u.uncertainty_cutoff_percent == 65

        p = models.Prediction(
            user_id=u.id,
            image_filename="file.jpg",
            class_predicted=1,
            confidence=0.9,
        )
        models.db.session.add(p)
        models.db.session.commit()

        assert p.id is not None
        assert p.user_id == u.id
        assert u.predictions[0].id == p.id


def test_duplicate_username_is_rejected_by_database(flask_app):
    """SQLite должен запретить второго пользователя с тем же username."""
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models

    with flask_app.app_context():
        u1 = models.User(username="same_name", email="first@example.com", password_hash="x")
        u2 = models.User(username="same_name", email="second@example.com", password_hash="y")
        models.db.session.add(u1)
        models.db.session.commit()
        models.db.session.add(u2)
        with pytest.raises(IntegrityError):
            models.db.session.commit()
        models.db.session.rollback()


def test_duplicate_email_is_rejected_by_database(flask_app):
    interface_dir = flask_app.template_folder.rsplit("templates", 1)[0].rstrip("\\/")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)

    import models

    with flask_app.app_context():
        u1 = models.User(username="u_a", email="shared@example.com", password_hash="x")
        u2 = models.User(username="u_b", email="shared@example.com", password_hash="y")
        models.db.session.add(u1)
        models.db.session.commit()
        models.db.session.add(u2)
        with pytest.raises(IntegrityError):
            models.db.session.commit()
        models.db.session.rollback()
