import os
import threading
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from database import init_db, get_user, add_request, get_requests, update_request
from Model.ResNet import CustomResNet

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomResNet(num_classes=2).to(device)
model.load_state_dict(torch.load('D:/nnModels/RGDetector/run_1/best_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(req_id, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            confidence = probabilities[predicted_class].item()
        result = "Real" if predicted_class == 0 else "Generated"
        update_request(req_id, 'completed', result, round(confidence * 100, 2))
    except Exception as e:
        update_request(req_id, 'error', str(e), 0)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    requests = get_requests(session['user_id'])
    return render_template('dashboard.html', requests=requests, username=session['username'])

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if 'file' not in request.files:
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('dashboard'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    req_id = add_request(session['user_id'], filepath)
    thread = threading.Thread(target=process_image, args=(req_id, filepath))
    thread.start()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    init_db()
    app.run(debug=True, port=5000)