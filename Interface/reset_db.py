import os
from app import app, db


def reset_database():
    # Сначала получаем путь к БД
    with app.app_context():
        db_path = os.path.join(app.instance_path, 'site.db')

        # Проверяем, существует ли файл
        if not os.path.exists(db_path):
            print("База данных не найдена. Создаю новую...")
            db.create_all()
            print("Новая база данных создана.")
            return

        # Закрываем все соединения внутри контекста
        db.session.remove()
        db.engine.dispose()

    # Выходим из контекста, чтобы освободить файл для удаления
    try:
        os.remove(db_path)
        print(f"Файл БД удалён: {db_path}")
    except PermissionError:
        print("ОШИБКА: Не удалось удалить файл базы данных.")
        print("Убедитесь, что Flask-приложение остановлено, и повторите попытку.")
        return
    except Exception as e:
        print(f"Неожиданная ошибка при удалении файла: {e}")
        return

    # Снова в контексте создаём новые таблицы
    with app.app_context():
        db.create_all()
        print("Новая база данных успешно создана.")


if __name__ == '__main__':
    reset_database()