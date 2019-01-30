from app import app
from learner.data_prep import *
import json
import pickle


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


@app.route('/predict')
def predict():
    sample_input = pd.DataFrame({
            'GRE': [320],
            'TOEFL': [120],
            'university_rating': [4],
            'statement_of_purpose': [4],
            'letter_of_recommendation': [4.5],
            'GPA': [9.65],
            'research': [1]
            })
    linear_model = pickle.load(open('models/linear_regression.sav', 'rb'))
    predict = linear_model.predict(sample_input)
    print(predict)
    return 'Worked!!!'
