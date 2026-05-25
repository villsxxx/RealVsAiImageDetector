# import sys
# from pathlib import Path
#
# import pytest
# from flask import Flask
# from flask_login import LoginManager
#
#
# @pytest.fixture(scope="session")
# def project_root():
#     return Path(__file__).resolve().parents[1]
#
#
# @pytest.fixture(scope="session")
# def interface_dir(project_root):
#     return project_root / "Interface"
#
#
# @pytest.fixture
# def flask_app(interface_dir, tmp_path):
#     """
#     Небольшое тестовое Flask-приложение.
#
#     Важно: мы НЕ импортируем `Interface/app.py`, чтобы тесты не подтягивали torch/модель.
#     """
#     if str(interface_dir) not in sys.path:
#         sys.path.insert(0, str(interface_dir))
#
#     import models  # noqa: E402
#
#     app = Flask(
#         __name__,
#         template_folder=str(interface_dir / "templates"),
#         static_folder=str(interface_dir / "static"),
#     )
#     app.config.update(
#         SECRET_KEY="test-secret",
#         TESTING=True,
#         WTF_CSRF_ENABLED=False,
#         SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'test.db'}",
#         SQLALCHEMY_TRACK_MODIFICATIONS=False,
#     )
#
#     models.db.init_app(app)
#
#     login_manager = LoginManager(app)
#
#     @login_manager.user_loader
#     def _load_user(user_id):
#         try:
#             return models.User.query.get(int(user_id))
#         except Exception:
#             return None
#
#     with app.app_context():
#         models.db.create_all()
#
#     return app