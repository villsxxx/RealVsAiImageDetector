import os
from flask import Flask, render_template, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from datetime import datetime
import uuid

from models import db, User, Prediction
from forms import LoginForm, RegisterForm
from predictor import predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-it'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    return render_template('dashboard.html', predictions=predictions)


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

    # Удаляем файл с диска (опционально)
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
        # Если ошибка, удаляем загруженный файл
        if os.path.exists(filepath):
            os.remove(filepath)
        flash(f'Ошибка при обработке: {str(e)}', 'danger')
        return redirect(url_for('index'))

    # Сохраняем только имя файла в БД
    prediction = Prediction(
        user_id=current_user.id,
        image_filename=filename,
        class_predicted=pred_class,
        confidence=confidence
    )
    db.session.add(prediction)
    db.session.commit()

    return redirect(url_for('prediction_detail', pred_id=prediction.id))


@app.route('/prediction/<int:pred_id>')
@login_required
def prediction_detail(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('У вас нет доступа к этой записи.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', pred=pred)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)