import os
import pandas as pd
import uuid
import joblib

model_filename = os.path.join(os.path.dirname(__file__), 'ml_model', 'bumrah', 'model.joblib')
data_filename = os.path.join(os.path.dirname(__file__), 'ml_model', 'bumrah', 'data.joblib')
transition_filename = os.path.join(os.path.dirname(__file__), 'ml_model', 'bumrah', 'transitions.joblib')
loaded_data = None
loaded_model = None
selected_features = None
transition_matrices = None

weight_prev_isBoundary = None
weight_innings_id = None

model_processor = None

pattern_transition_matrices = None

file_path = os.path.join(os.path.dirname(__file__), 'dataset', 'bumrah_test.csv')

data = []


def get_data():
    global data
    global loaded_data
    global loaded_model
    global selected_features
    global transition_matrices
    global weight_prev_isBoundary
    global weight_innings_id
    global model_processor
    global pattern_transition_matrices

    data = pd.read_csv(file_path)
    data = filter_data()

    loaded_model = joblib.load(model_filename)
    loaded_data = joblib.load(data_filename)
    loaded_transitions = joblib.load(transition_filename)
    selected_features = loaded_data['selected_features']
    transition_matrices = loaded_data['transition_matrices']

    weight_prev_isBoundary = loaded_data['weight_prev_isBoundary']
    weight_innings_id = loaded_data['weight_innings_id']

    model_processor = loaded_data['model_processor']

    pattern_transition_matrices = loaded_transitions['transition_matrices']


def filter_data():
    global data
    # Generate and add unique IDs as a new column
    data['id'] = [str(uuid.uuid4()) for _ in range(len(data))]
    dataset = data.tail(12)
    return dataset


get_data()
