from app import app
from learner.data_prep import *
@app.route('/')
def index():
    csv_path = 'data/Admission_Predict.csv'
    dataframe = prep_data(csv_path)
    print(dataframe.columns)
