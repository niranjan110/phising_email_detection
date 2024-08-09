from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
import urllib.parse

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# MongoDB Atlas connection
username = urllib.parse.quote_plus("niranjanreddyr2002")
password = urllib.parse.quote_plus("2002@niranjan")
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.a73fx.mongodb.net/email_phishing_detection?retryWrites=true&w=majority")
db = client['email_phishing_detection']
users_collection = db['users']

# Load the model and vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('vectorizer.pkl')

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({'username': username}):
            flash('Username already exists! Please choose a different one.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        email_text = request.form['email']
        if not email_text.strip():
            flash('Email content cannot be empty.', 'danger')
            return redirect(url_for('dashboard'))

        email_vector = tfidf.transform([email_text])
        prediction = model.predict(email_vector)[0]
        prediction_proba = model.predict_proba(email_vector)[0][1]  # Get phishing probability
        print(prediction)
        is_spam = prediction == "Safe Email"
        print(is_spam)
        result = 'Phishing' if is_spam!=1 else 'Safe'
        spam_percentage = prediction_proba * 100
        return render_template('result.html', result=result, spam_percentage=spam_percentage)

    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
