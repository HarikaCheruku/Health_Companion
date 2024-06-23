from flask import Flask, render_template, redirect, url_for, session, flash, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
import bcrypt
import sqlite3
import os
from gtts import gTTS
from flask import send_file
from chat import classify_intent, get_response
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='assets')
app.secret_key = '1234'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE = 'health_companion.db'

def get_db():
    db_path = os.path.join(BASE_DIR, 'health_companion.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


from wtforms.validators import ValidationError


def validate_email_format(form, field):
    email = field.data
    if not (email.endswith('@gmail.com') or email.endswith('@outlook.com')):
        raise ValidationError('Invalid email format. Please use a valid Gmail or Outlook email address.')

class RegistrationForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email(), validate_email_format])
    password = PasswordField("Password", validators=[DataRequired()])
    submit_register = SubmitField("Register")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), validate_email_format])
    password = PasswordField("Password", validators=[DataRequired()])
    submit_login = SubmitField("Login")

class ForgotPasswordForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), validate_email_format])
    new_password = PasswordField("New Password", validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo('new_password')])
    submit_reset_password = SubmitField("Update Password")

@app.route('/')
def index():
    return redirect(url_for('login'))
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM login1 WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and password == user['password']:
            session['user_id'] = user['id']
            return redirect(url_for('home'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)



@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data  # No hashing

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO login1 (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        conn.close()

        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        new_password = form.new_password.data

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM login1 WHERE email = ?", (email,))
        account = cursor.fetchone()

        if account:
            cursor.execute("UPDATE login1 SET password = ? WHERE email = ?", (new_password, email))  # Store plain text new password
            conn.commit()
            conn.close()

            flash("Password updated successfully. Please log in with your new password.", "success")
            return redirect(url_for('login'))
        else:
            conn.close()
            flash("The email does not exist.", "danger")

    return render_template('forgot_password.html', form=form)


with open(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\intents.json") as json_file:
    intents = json.load(json_file)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\label_encoder_functional.npy")
vectorizer = CountVectorizer()
vectorizer.vocabulary_ = np.load(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\vectorizer_functional.npy", allow_pickle=True).item()


model = load_model(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\intent_classification_model_functional.h5")


rfc = pickle.load(open(r"C:\Users\HP\PycharmProjects\Health Companion\models\rfc.pkl", 'rb'))



precautions = pd.read_csv(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\precautions_df.csv")
workout = pd.read_csv(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\workout_df.csv")
description = pd.read_csv(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\description.csv")
medications = pd.read_csv(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\medications.csv")
diets = pd.read_csv(r"C:\Users\HP\PycharmProjects\Health Companion\datasets\precautions_df.csv")

import ast
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].iloc[0]

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist()

    med = medications[medications['Disease'] == dis]['Medication'].iloc[0]
    med = ast.literal_eval(med)

    die = diets[diets['Disease'] == dis]['Diet'].iloc[0]
    die = ast.literal_eval(die)

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[rfc.predict([input_vector])[0]]
@app.route('/symptoms_input')
def symptoms_input():
    return render_template('symptoms_input.html', symptoms_dict=symptoms_dict)



import requests
import ast
from flask import render_template

def fetch_image_urls(diet_names):
    unsplash_access_key = 'mtFEnOXGJ6eX1AVuZa6LgKDZloxvJyNxtNswm3FOeQ4'
    image_urls = {}

    for diet_name in diet_names:
        response = requests.get(
            'https://api.unsplash.com/search/photos',
            params={'query': diet_name},
            headers={'Authorization': f'Client-ID {unsplash_access_key}'}
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                image_url = data['results'][0]['urls']['regular']
                image_urls[diet_name] = image_url

    return image_urls

@app.route('/predict', methods=['POST'])
def predict_disease():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        predicted_disease = get_predicted_value(symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        diet_image_urls = fetch_image_urls(rec_diet)

        return render_template('prediction.html',
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout,
                               diet_image_urls=diet_image_urls)

@app.route('/speak_aloud', methods=['POST'])
def speak_aloud():
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('lang', 'en')

    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")

    return send_file("output.mp3", as_attachment=False)


@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    message = request.form['message']
    if message.lower() == 'quit':
        return "Goodbye!"
    elif message.strip() == "":
        return "Please enter a valid input."

    intent, confidence = classify_intent(message)
    if confidence <= 0.5:
        return "Sorry, I can't help you with that."

    response = get_response(intent)
    return response


if __name__ == '__main__':
    with app.app_context():
        conn = get_db()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS login1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        ''')

        # Insert a test user if not exists
        cursor.execute('''
        INSERT OR IGNORE INTO login1 (id, name, email, password) 
        VALUES (1, 'user1', 'user1@gmail.com', '1234')
        ''')

        conn.commit()
        conn.close()

    app.run(host='0.0.0.0', port=5000)