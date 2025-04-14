from flask import Flask, request, render_template, redirect, url_for, session, jsonify, flash
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

USER_FILE = 'users.json'


def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

try:
    model = load_model("bot_detector.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    model, tokenizer = None, None

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', str(text))
    text = re.sub(r'\W', ' ', text)
    text = text.lower().strip()
    return text

def predict_text(text):
    if not model or not tokenizer:
        return "Error: Model not loaded"
    cleaned_text = clean_text(text)
    sequence = pad_sequences(tokenizer.texts_to_sequences([cleaned_text]), maxlen=50)
    prediction = model.predict(sequence)
    return "Human" if prediction[0][0] > 0.5 else "Bot"

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    users = load_users()
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        number = request.form.get('number')
        address = request.form.get('address')
        
        if not all([username, email, password, number, address]):
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))

        if username in users:
            flash('User already exists! Please log in.', 'danger')
            return redirect(url_for('login'))

        users[username] = {
            'email': email,
            'password': generate_password_hash(password),
            'number': number,
            'address': address
        }
        save_users(users)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return redirect(url_for('login'))

        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route("/index")
def index():
    if 'username' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Please enter text to analyze.", "warning")
            return redirect(url_for("index"))
        result = predict_text(text)
        return render_template("index.html", result=result)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
