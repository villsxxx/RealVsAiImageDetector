import os
import tempfile
import requests
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from datetime import datetime
import uuid
import cv2
import numpy as np

from models import db, User, Prediction, BatchPrediction, FramePrediction
from forms import LoginForm, RegisterForm
from predictor import predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-it'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB для видео

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def download_image_from_url(url):
    """Скачивает изображение по URL и сохраняет во временный файл"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()

        # Определяем расширение из Content-Type или URL
        content_type = response.headers.get('content-type', '')
        ext = 'jpg'
        if 'png' in content_type:
            ext = 'png'
        elif 'gif' in content_type:
            ext = 'gif'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            ext = 'jpg'

        # Создаём временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()

        return temp_file.name
    except Exception as e:
        raise Exception(f"Не удалось загрузить изображение: {str(e)}")


def process_video_frames(video_path, sample_rate=1):
    """
    Обрабатывает видео, извлекая кадры с заданной частотой
    sample_rate: извлекать каждый N-ый кадр
    """
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Извлекаем каждый sample_rate-ый кадр
        if frame_count % sample_rate == 0:
            # Сохраняем кадр во временный файл
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, frame)

            try:
                # Предсказание для кадра
                pred_class, confidence = predictor.predict(temp_file.name)
                frame_predictions.append({
                    'frame_number': frame_count,
                    'class_predicted': pred_class,
                    'confidence': confidence,
                    'temp_path': temp_file.name
                })
                processed_frames += 1
            except Exception as e:
                print(f"Ошибка обработки кадра {frame_count}: {e}")
            finally:
                # Не удаляем временный файл сразу, он понадобится для отображения
                pass

        frame_count += 1

    cap.release()

    # Рассчитываем статистику
    if frame_predictions:
        ai_count = sum(1 for p in frame_predictions if p['class_predicted'] == 1)
        real_count = sum(1 for p in frame_predictions if p['class_predicted'] == 0)
        avg_confidence_ai = sum(
            p['confidence'] for p in frame_predictions if p['class_predicted'] == 1) / ai_count if ai_count > 0 else 0
        avg_confidence_real = sum((1 - p['confidence']) for p in frame_predictions if
                                  p['class_predicted'] == 0) / real_count if real_count > 0 else 0

        # Общий вердикт: большинство кадров
        total_frames_processed = len(frame_predictions)
        ai_percentage = (ai_count / total_frames_processed) * 100

        # Средняя уверенность по всем кадрам
        overall_confidence = sum(
            p['confidence'] if p['class_predicted'] == 1 else (1 - p['confidence'])
            for p in frame_predictions
        ) / total_frames_processed

        final_class = 1 if ai_count > real_count else 0
        is_uncertain = overall_confidence < 0.65

        return {
            'frame_predictions': frame_predictions,
            'statistics': {
                'total_frames_processed': total_frames_processed,
                'ai_frames': ai_count,
                'real_frames': real_count,
                'ai_percentage': ai_percentage,
                'avg_confidence_ai': avg_confidence_ai,
                'avg_confidence_real': avg_confidence_real,
                'overall_confidence': overall_confidence,
                'final_class': final_class,
                'is_uncertain': is_uncertain
            }
        }
    else:
        return None


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
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(
        10).all()
    batch_predictions = BatchPrediction.query.filter_by(user_id=current_user.id).order_by(
        BatchPrediction.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', predictions=predictions, batch_predictions=batch_predictions)


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    pagination = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=20)
    return render_template('history.html', pagination=pagination)


@app.route('/batch_history')
@login_required
def batch_history():
    page = request.args.get('page', 1, type=int)
    pagination = BatchPrediction.query.filter_by(user_id=current_user.id).order_by(
        BatchPrediction.timestamp.desc()).paginate(
        page=page, per_page=10)
    return render_template('batch_history.html', pagination=pagination)


@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Вся история удалена.', 'success')
    return redirect(url_for('history'))


@app.route('/clear_batch_history', methods=['POST'])
@login_required
def clear_batch_history():
    BatchPrediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Вся история пакетных проверок удалена.', 'success')
    return redirect(url_for('batch_history'))


@app.route('/delete_prediction/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('history'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], pred.image_filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.session.delete(pred)
    db.session.commit()
    flash('Запись удалена.', 'success')
    return redirect(url_for('history'))


@app.route('/delete_batch/<int:batch_id>', methods=['POST'])
@login_required
def delete_batch(batch_id):
    batch = BatchPrediction.query.get_or_404(batch_id)
    if batch.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('batch_history'))

    # Удаляем связанные кадры и файлы
    for frame in batch.frames:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], frame.image_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.session.delete(frame)

    db.session.delete(batch)
    db.session.commit()
    flash('Пакетная проверка удалена.', 'success')
    return redirect(url_for('batch_history'))


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    # Проверяем тип загрузки
    upload_type = request.form.get('upload_type', 'single')

    if upload_type == 'single':
        return upload_single_image()
    elif upload_type == 'multiple':
        return upload_multiple_images()
    elif upload_type == 'url':
        return upload_from_url()
    elif upload_type == 'video':
        return upload_video()
    else:
        flash('Неизвестный тип загрузки', 'danger')
        return redirect(url_for('index'))


def upload_single_image():
    if 'file' not in request.files:
        flash('Файл не найден', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('Файл не выбран', 'danger')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Недопустимый формат файла', 'danger')
        return redirect(url_for('index'))

    # Генерируем уникальное имя файла
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        pred_class, confidence = predictor.predict(filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        flash(f'Ошибка при обработке: {str(e)}', 'danger')
        return redirect(url_for('index'))

    prediction = Prediction(
        user_id=current_user.id,
        image_filename=filename,
        class_predicted=pred_class,
        confidence=confidence
    )
    db.session.add(prediction)
    db.session.commit()

    return redirect(url_for('prediction_detail', pred_id=prediction.id))


def upload_multiple_images():
    if 'files' not in request.files:
        flash('Файлы не найдены', 'danger')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('Файлы не выбраны', 'danger')
        return redirect(url_for('index'))

    predictions = []
    for file in files:
        if not allowed_file(file.filename):
            continue

        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            pred_class, confidence = predictor.predict(filepath)
            prediction = Prediction(
                user_id=current_user.id,
                image_filename=filename,
                class_predicted=pred_class,
                confidence=confidence
            )
            db.session.add(prediction)
            predictions.append(prediction)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Ошибка при обработке {file.filename}: {str(e)}', 'danger')

    db.session.commit()

    if predictions:
        flash(f'Успешно обработано {len(predictions)} изображений', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Не удалось обработать ни одного изображения', 'danger')
        return redirect(url_for('index'))


def upload_from_url():
    url = request.form.get('url', '').strip()
    if not url:
        flash('URL не указан', 'danger')
        return redirect(url_for('index'))

    try:
        temp_path = download_image_from_url(url)

        # Генерируем имя для сохранения
        ext = temp_path.split('.')[-1]
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Перемещаем временный файл в постоянное хранилище
        import shutil
        shutil.move(temp_path, filepath)

        pred_class, confidence = predictor.predict(filepath)

        prediction = Prediction(
            user_id=current_user.id,
            image_filename=filename,
            class_predicted=pred_class,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()

        return redirect(url_for('prediction_detail', pred_id=prediction.id))

    except Exception as e:
        flash(f'Ошибка при загрузке по URL: {str(e)}', 'danger')
        return redirect(url_for('index'))


def upload_video():
    if 'video' not in request.files:
        flash('Видеофайл не найден', 'danger')
        return redirect(url_for('index'))

    video = request.files['video']
    if video.filename == '':
        flash('Файл не выбран', 'danger')
        return redirect(url_for('index'))

    if not allowed_video(video.filename):
        flash('Недопустимый формат видеофайла', 'danger')
        return redirect(url_for('index'))

    # Сохраняем видео во временный файл
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.rsplit('.', 1)[1].lower()}")
    video.save(temp_video.name)
    temp_video.close()

    # Получаем параметры обработки
    sample_rate = int(request.form.get('sample_rate', 5))  # Каждый N-ый кадр
    max_frames = int(request.form.get('max_frames', 100))  # Максимум кадров для обработки

    try:
        # Обрабатываем видео
        result = process_video_frames(temp_video.name, sample_rate)

        if not result:
            flash('Не удалось извлечь кадры из видео', 'danger')
            os.unlink(temp_video.name)
            return redirect(url_for('index'))

        # Создаём запись пакетной проверки
        batch = BatchPrediction(
            user_id=current_user.id,
            batch_type='video',
            source_filename=video.filename,
            total_items=result['statistics']['total_frames_processed'],
            summary_stats={
                'ai_frames': result['statistics']['ai_frames'],
                'real_frames': result['statistics']['real_frames'],
                'ai_percentage': result['statistics']['ai_percentage'],
                'overall_confidence': result['statistics']['overall_confidence'],
                'final_class': result['statistics']['final_class'],
                'is_uncertain': result['statistics']['is_uncertain']
            }
        )
        db.session.add(batch)
        db.session.flush()  # Получаем batch.id

        # Сохраняем кадры
        for frame_data in result['frame_predictions']:
            # Перемещаем временный файл кадра в постоянное хранилище
            frame_filename = f"{uuid.uuid4().hex}.jpg"
            frame_filepath = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
            import shutil
            shutil.move(frame_data['temp_path'], frame_filepath)

            frame_pred = FramePrediction(
                batch_id=batch.id,
                user_id=current_user.id,
                image_filename=frame_filename,
                frame_number=frame_data['frame_number'],
                class_predicted=frame_data['class_predicted'],
                confidence=frame_data['confidence']
            )
            db.session.add(frame_pred)

        db.session.commit()

        # Удаляем временный видеофайл
        os.unlink(temp_video.name)

        flash(f'Видео обработано. Проанализировано {result["statistics"]["total_frames_processed"]} кадров.', 'success')
        return redirect(url_for('batch_detail', batch_id=batch.id))

    except Exception as e:
        flash(f'Ошибка при обработке видео: {str(e)}', 'danger')
        if os.path.exists(temp_video.name):
            os.unlink(temp_video.name)
        return redirect(url_for('index'))


@app.route('/prediction/<int:pred_id>')
@login_required
def prediction_detail(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', pred=pred)


@app.route('/batch/<int:batch_id>')
@login_required
def batch_detail(batch_id):
    batch = BatchPrediction.query.get_or_404(batch_id)
    if batch.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('dashboard'))

    frames = FramePrediction.query.filter_by(batch_id=batch_id).order_by(FramePrediction.frame_number).all()
    return render_template('batch_result.html', batch=batch, frames=frames)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)