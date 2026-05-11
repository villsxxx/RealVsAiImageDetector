import sys


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

