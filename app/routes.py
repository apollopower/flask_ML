from app import app
from learner.data_prep import *
import json
@app.route('/')
def index():
    return "Hello there young Snorlax"

@app.route('/train')
def train():
    data_path = 'data/Admission_Predict.csv'
    features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
    target = 'Chance of Admit '
    data_map = create_supervised_dataset(data_path, features, target)
    y_predict = linear_regression(data_map)
    return json.dumps(y_predict)
