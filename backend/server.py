from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import os
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GRAPH_FOLDER = 'input_graphs'
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Ensure the UPLOAD_FOLDER exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'success': True, 'filename': filename})

@app.route("/files", methods=['GET'])
def get_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'files': files})

admin_credentials = {'email': 'admin@example.com', 'password': 'admin123'}
hr_credentials = {'email': 'hr@example.com', 'password': 'hr123'}

def create_neural_network(input_shape, learning_rate=0.1):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)  # Set the learning rate here
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if email == admin_credentials['email'] and password == admin_credentials['password']:
        return jsonify({'success': True, 'message': 'Admin login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/hr/login', methods=['POST'])
def hr_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if email == hr_credentials['email'] and password == hr_credentials['password']:
        return jsonify({'success': True, 'message': 'HR login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})


@app.route("/")
def hello():
    return "hello world"

# Define fuzzy logic control system
def define_control_system():
    # Input variables
    tenth = ctrl.Antecedent(np.arange(0.0, 101.0, 1), 'tenth')
    twelfth = ctrl.Antecedent(np.arange(0.0, 101.0, 1), 'twelfth')
    graduation = ctrl.Antecedent(np.arange(0.0, 101.0, 1), 'graduation')
    years_experience = ctrl.Antecedent(np.arange(0.0, 21.0, 1), 'years_experience')
    
    # Output
    result = ctrl.Consequent(np.arange(0.0, 101.0, 1), 'result')

    # Membership functions
    tenth['low'] = fuzz.gaussmf(tenth.universe, 0, 20)
    tenth['medium'] = fuzz.gaussmf(tenth.universe, 50, 20)
    tenth['high'] = fuzz.gaussmf(tenth.universe, 100, 20)
    
    twelfth['low'] = fuzz.gaussmf(twelfth.universe, 0, 20)
    twelfth['medium'] = fuzz.gaussmf(twelfth.universe, 50, 20)
    twelfth['high'] = fuzz.gaussmf(twelfth.universe, 100, 20)
    
    graduation['low'] = fuzz.gaussmf(graduation.universe, 0, 20)
    graduation['medium'] = fuzz.gaussmf(graduation.universe, 50, 20)
    graduation['high'] = fuzz.gaussmf(graduation.universe, 100, 20)
    
    years_experience['low'] = fuzz.gaussmf(years_experience.universe, 0, 5)
    years_experience['medium'] = fuzz.gaussmf(years_experience.universe, 10, 5)
    years_experience['high'] = fuzz.gaussmf(years_experience.universe, 20, 5)
    
    result['low'] = fuzz.gaussmf(result.universe, 0, 20)
    result['medium'] = fuzz.gaussmf(result.universe, 50, 20)
    result['high'] = fuzz.gaussmf(result.universe, 100, 20)

    for var in [tenth, twelfth, graduation, years_experience, result]:
        var.view()
        plt.savefig(os.path.join(GRAPH_FOLDER, f'{var.label}_graph.png'))
        plt.close()

    # Rules
    rules = []
    for tenth_level in ['low', 'medium', 'high']:
        for twelfth_level in ['low', 'medium', 'high']:
            for grad_level in ['low', 'medium', 'high']:
                for exp_level in ['low', 'medium', 'high']:
                    for res_level in ['low', 'medium', 'high']:
                        rule = ctrl.Rule(tenth[tenth_level] & twelfth[twelfth_level] & graduation[grad_level] & years_experience[exp_level], result[res_level])
                        rules.append(rule)

    result_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(result_ctrl)

# Load fuzzy logic control system
results = define_control_system()

def get_label(value):
    if value >= 50:
        return 'Excellent'
    elif value >= 30:
        return 'Good'
    elif value >= 20:
        return 'Fair'
    else:
        return 'Poor'

def convert_to_gaussian(value, mu, sigma):
    gaussian_value = np.exp(-0.5 * ((value - mu) / sigma) ** 2)
    return gaussian_value


@app.route("/input", methods=['POST'])
def input():
    data = request.json
    file_name = data.get('selectedFile', '')
    if not file_name:
        return jsonify({'error': 'No file selected'})
    tenth = float(data.get("tenth"))
    twelth = float(data.get("twelth"))
    grad = float(data.get("grad"))
    yoe = float(data.get("yoe"))
    required_skills = data.get("skills", [])
    sort_criteria = data.get("sortCriteria", "result")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    filtered_data = filter_data(file_path, tenth, twelth, grad, yoe)
    output, accuracy, precision, recall, f1_score = compute_results(filtered_data, required_skills, sort_criteria)
    return jsonify({
        'results': output,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    })

def train_nn_model(file_path, test_size=0.2):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tenth_percentage = float(row['tenth_percentage'])
            twelfth_percentage = float(row['twelfth_percentage'])
            graduation_percentage = float(row['graduation_percentage'])
            years_experience = float(row['years_experience'])
            data.append((tenth_percentage, twelfth_percentage, graduation_percentage, years_experience))

    X = []  # Input features
    y = []  # Target labels

    for row in data:
        tenth_percentage, twelfth_percentage, graduation_percentage, years_experience = row
        results.input['tenth'] = tenth_percentage
        results.input['twelfth'] = twelfth_percentage
        results.input['graduation'] = graduation_percentage
        results.input['years_experience'] = years_experience
        results.compute()
        
        # Extract fuzzy logic output as result
        result = results.output['result']

        # Convert features to Gaussian values
        #tenth_gaussian = max(convert_to_gaussian(tenth_percentage, 0, 20), convert_to_gaussian(tenth_percentage, 50, 20), convert_to_gaussian(tenth_percentage, 100, 20))
        #twelfth_gaussian = max(convert_to_gaussian(twelfth_percentage, 0, 20), convert_to_gaussian(twelfth_percentage, 50, 20), convert_to_gaussian(twelfth_percentage, 100, 20))
        #grad_gaussian = max(convert_to_gaussian(graduation_percentage, 0, 20), convert_to_gaussian(graduation_percentage, 50, 20), convert_to_gaussian(graduation_percentage, 100, 20))
        #yoe_gaussian = max(convert_to_gaussian(years_experience, 0, 5), convert_to_gaussian(years_experience, 10, 5), convert_to_gaussian(years_experience,20, 5))
        
        # Append features and result to X and y
        X.append([tenth_percentage,twelfth_percentage,graduation_percentage,years_experience])
        y.append(result)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Define and train the neural network model using X_train and y_train
    model = create_neural_network(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)

    return accuracy


def filter_data(file_path, min_tenth, min_twelfth, min_graduation, yoe):
    data = []
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tenth_percentage = float(row['tenth_percentage'])
            twelfth_percentage = float(row['twelfth_percentage'])
            graduation_percentage = float(row['graduation_percentage'])
            education = row['education']
            years_experience = float(row['years_experience'])
            label = row['Criteria_Label']
            education_level = education.split(" ")[0].strip().lower()
            if (tenth_percentage >= min_tenth and
                twelfth_percentage >= min_twelfth and
                graduation_percentage >= min_graduation and years_experience >= yoe):
                name = row['name']
                skills = row['skills'].lower()
                data.append((name, tenth_percentage, twelfth_percentage, graduation_percentage, education_level, skills, years_experience, label))
    return data




def compute_results(data, required_skills, sort_criteria, save_confusion_matrix=True, save_dir='confusion_matrices'):
    output = []
    true_positives = 0
    false_positives = 0
    
    for row in data:
        name, tenth_percentage, twelfth_percentage, graduation_percentage, education_level, skills, years_experience, label = row
        results.input['tenth'] = tenth_percentage
        results.input['twelfth'] = twelfth_percentage
        results.input['graduation'] = graduation_percentage
        results.input['years_experience'] = years_experience
        results.compute()
        skills_list = skills.split(", ")
        required_skills_str = " ".join(required_skills)
        vectorizer = CountVectorizer()
        skills_vector = vectorizer.fit_transform([" ".join(skills_list)])
        required_skills_vector = vectorizer.transform([required_skills_str])
        cosine_sim = cosine_similarity(skills_vector, required_skills_vector)
        
        output.append({
            'name': name,
            'tenth_percentage': tenth_percentage,
            'twelfth_percentage': twelfth_percentage,
            'graduation_percentage': graduation_percentage,
            'education': education_level,
            'years_of_experience': years_experience,
            'result': (results.output['result']),
            'matching_skills_percentage': cosine_sim[0][0] * 100,
            'label': label
        })

        if label == "1":
            true_positives += 1
        else:
            false_positives += 1
    
    total_records = len(output)
    
    accuracy = true_positives / total_records if total_records > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_records if total_records > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Generate confusion matrix
    cm = [[0, 0],[false_positives,true_positives]]
    
    # Save confusion matrix
    if save_confusion_matrix:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

    if sort_criteria == "composite_score":
        for record in output:
            record['composite_score'] = record['result'] + record['matching_skills_percentage'] + record['tenth_percentage'] + record['twelfth_percentage'] + record['graduation_percentage'] + record['years_of_experience']
        output_sorted = sorted(output, key=lambda x: x['composite_score'], reverse=True)
    else:
        output_sorted = sorted(output, key=lambda x: x[sort_criteria], reverse=True)

    return output_sorted, accuracy * 100, precision, recall, f1_score


if __name__ == "__main__":
    app.run(debug=True)
