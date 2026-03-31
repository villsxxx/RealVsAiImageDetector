import os
import threading
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from datetime import datetime
import uuid
import tempfile
import shutil

from models import db, User, Prediction
from forms import LoginForm, RegisterForm
from predictor import predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-it'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
# Видео обычно весит больше картинок, поэтому лимит увеличиваем
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

# Хранилище задач обработки видео (упрощённо: в памяти процесса)
# Важно: если перезапустить сервер — задачи сотрутся.
VIDEO_TASKS = {}
VIDEO_TASKS_LOCK = threading.Lock()


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('Слишком большой объём данных. Попробуйте загрузить меньше файлов или уменьшить размер.', 'danger')
    return redirect(url_for('index'))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mpeg', 'mpg'}


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Регистрация прошла успешно!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            flash('Вы успешно вошли!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Неверное имя пользователя или пароль', 'danger')
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Вы вышли из системы.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Для карточек показываем последние проверки, но статистику считаем по всем.
    recent_predictions = (
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
    days_activity = ((datetime.utcnow() - first_pred.timestamp).days) + 1 if first_pred else 0

    return render_template(
        'dashboard.html',
        recent_predictions=recent_predictions,
        stats_total=stats_total,
        stats_real=stats_real,
        stats_ai=stats_ai,
        days_activity=days_activity
    )


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    pagination = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=20)
    return render_template('history.html', pagination=pagination)


@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Вся история удалена.', 'success')
    return redirect(url_for('history'))


@app.route('/delete_prediction/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('history'))

    # Удаляем файл с диска
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], pred.image_filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.session.delete(pred)
    db.session.commit()
    flash('Запись удалена.', 'success')
    return redirect(url_for('history'))


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    user_id = current_user.id
    image_files = request.files.getlist('files')
    video_file = request.files.get('video')

    has_images = any(f and f.filename for f in image_files)
    has_video = bool(video_file and video_file.filename)

    if not has_images and not has_video:
        flash('Выберите изображения или видео', 'danger')
        return redirect(url_for('index'))

    # Сохраняем изображения заранее (чтобы фоновой задаче было с чем работать)
    saved_image_info = []  # (filepath_in_uploads, filename)
    if has_images:
        for file in image_files:
            if not file or not file.filename:
                continue
            if not allowed_image_file(file.filename):
                flash('Недопустимый формат изображения', 'danger')
                return redirect(url_for('index'))

            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_image_info.append((filepath, filename))

    # Если видео нет — можно обработать синхронно (как раньше)
    if not has_video:
        if not saved_image_info:
            flash('Нечего анализировать: файлы не выбраны', 'danger')
            return redirect(url_for('index'))

        created_predictions = []
        try:
            for filepath, filename in saved_image_info:
                pred_class, confidence = predictor.predict(filepath)
                created_predictions.append(
                    Prediction(
                        user_id=user_id,
                        image_filename=filename,
                        class_predicted=pred_class,
                        confidence=confidence
                    )
                )
            for p in created_predictions:
                db.session.add(p)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            for filepath, _ in saved_image_info:
                if os.path.exists(filepath):
                    os.remove(filepath)
            flash(f'Ошибка при обработке: {str(e)}', 'danger')
            return redirect(url_for('index'))

        if len(created_predictions) == 1:
            return redirect(url_for('prediction_detail', pred_id=created_predictions[0].id))
        return redirect(url_for('dashboard'))

    # Видео есть — запускаем фоную задачу с прогрессом
    if not allowed_video_file(video_file.filename):
        flash('Недопустимый формат видео', 'danger')
        return redirect(url_for('index'))

    video_ext = video_file.filename.rsplit('.', 1)[1].lower()
    video_tmp_dir = tempfile.mkdtemp(prefix="upload_video_")
    video_tmp_path = os.path.join(video_tmp_dir, f"input.{video_ext}")
    video_file.save(video_tmp_path)

    task_id = uuid.uuid4().hex

    with VIDEO_TASKS_LOCK:
        VIDEO_TASKS[task_id] = {
            "user_id": user_id,
            "status": "queued",
            "processed_frames": 0,
            "total_frames": 0,
            "done": False,
            "prediction_ids": [],
            "error": None,
        }

    def process_task():
        preview_filename = None
        preview_path = None
        created_predictions = []
        video_prediction = None
        try:
            with app.app_context():
                with VIDEO_TASKS_LOCK:
                    if task_id in VIDEO_TASKS:
                        VIDEO_TASKS[task_id]["status"] = "running"

                # 1) Изображения (без прогресса)
                for filepath, filename in saved_image_info:
                    pred_class, confidence = predictor.predict(filepath)
                    created_predictions.append(
                        Prediction(
                            user_id=user_id,
                            image_filename=filename,
                            class_predicted=pred_class,
                            confidence=confidence
                        )
                    )

                # 2) Видео (с прогрессом)
                preview_filename = f"{uuid.uuid4().hex}.jpg"
                preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)

                def on_frame_processed(processed, total):
                    with VIDEO_TASKS_LOCK:
                        task = VIDEO_TASKS.get(task_id)
                        if not task:
                            return
                        task["processed_frames"] = int(processed)
                        task["total_frames"] = int(total)

                pred_class, confidence = predictor.predict(
                    video_tmp_path,
                    preview_out_path=preview_path,
                    video_progress_callback=on_frame_processed
                )
                video_prediction = Prediction(
                    user_id=user_id,
                    image_filename=preview_filename,
                    class_predicted=pred_class,
                    confidence=confidence
                )
                created_predictions.append(video_prediction)

                for p in created_predictions:
                    db.session.add(p)
                db.session.commit()

                pred_ids = [p.id for p in created_predictions]
                video_pred_id = video_prediction.id if video_prediction else None

                with VIDEO_TASKS_LOCK:
                    task = VIDEO_TASKS.get(task_id)
                    if task:
                        task["prediction_ids"] = pred_ids
                        task["video_prediction_id"] = video_pred_id
                        task["done"] = True
                        task["status"] = "finished"

        except Exception as e:
            with app.app_context():
                db.session.rollback()
            # Удаляем сохранённые файлы изображений
            for filepath, _ in saved_image_info:
                if os.path.exists(filepath):
                    os.remove(filepath)
            # Удаляем превью видео
            if preview_path and os.path.exists(preview_path):
                os.remove(preview_path)

            with VIDEO_TASKS_LOCK:
                task = VIDEO_TASKS.get(task_id)
                if task:
                    task["error"] = str(e)
                    task["done"] = True
                    task["status"] = "error"

        finally:
            shutil.rmtree(video_tmp_dir, ignore_errors=True)

    threading.Thread(target=process_task, daemon=True).start()

    return jsonify({"task_id": task_id}), 202


@app.route('/prediction/<int:pred_id>')
@login_required
def prediction_detail(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', pred=pred)


@app.route('/profile')
@login_required
def profile():
    # Получаем все предсказания пользователя
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    # Считаем статистику
    predictions_count = len(predictions)
    real_count = sum(1 for p in predictions if p.class_predicted == 0)
    ai_count = sum(1 for p in predictions if p.class_predicted == 1)

    # Средняя уверенность
    if predictions_count > 0:
        total_confidence = sum(p.confidence for p in predictions)
        avg_confidence = total_confidence / predictions_count
    else:
        avg_confidence = 0

    return render_template('profile.html',
                           predictions_count=predictions_count,
                           real_count=real_count,
                           ai_count=ai_count,
                           avg_confidence=avg_confidence)


@app.route('/api/user_stats')
@login_required
def user_stats():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    predictions_count = len(predictions)
    real_count = sum(1 for p in predictions if p.class_predicted == 0)
    ai_count = sum(1 for p in predictions if p.class_predicted == 1)

    if predictions_count > 0:
        avg_confidence = sum(p.confidence for p in predictions) / predictions_count
    else:
        avg_confidence = 0

    return {
        'total': predictions_count,
        'real': real_count,
        'ai': ai_count,
        'avg_confidence': round(avg_confidence * 100, 2)
    }


@app.route('/api/video_task/<task_id>', methods=['GET'])
@login_required
def video_task_status(task_id):
    with VIDEO_TASKS_LOCK:
        task = VIDEO_TASKS.get(task_id)

    if not task:
        return jsonify({"error": "task not found"}), 404
    if task["user_id"] != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    return jsonify({
        "task_id": task_id,
        "status": task.get("status"),
        "processed_frames": task.get("processed_frames", 0),
        "total_frames": task.get("total_frames", 0),
        "done": task.get("done", False),
        "prediction_ids": task.get("prediction_ids", []),
        "video_prediction_id": task.get("video_prediction_id"),
        "error": task.get("error"),
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)