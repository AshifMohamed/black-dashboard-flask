import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from apps.home.transitions import calculate_probable_length
from apps.data_loader import selected_features, loaded_model, data, model_processor

label_encoder_org = model_processor['label_encoder']

scaler_org = model_processor['scaler']

# List of all possible categories for the 'previous_ball_length' feature
all_prev_ball_lengths = ['First ball', 'Short of length', 'Full', 'Short', 'Good length', 'Full toss', 'Yorker']
all_cluster_names = ['Balanced', 'Aggressive', 'Bowler']

categorical_features = ['prev_ball_length', 'cluster_name']

all_numerical_features = ['over', 'bowler1_wkts', 'runs_before_ball', 'wickets_before_ball',
                          'required_rate', 'batsman1_runs_before_ball', 'required_rate_bin']

all_numerical_features_scaled = ['over_scaled', 'bowler1_wkts_scaled', 'runs_before_ball_scaled',
                                 'wickets_before_ball_scaled',
                                 'required_rate_scaled', 'batsman1_runs_before_ball_scaled', 'required_rate_bin_scaled']

all_categorical_features_encoded = ['prev_ball_length_First ball', 'prev_ball_length_Full',
                                    'prev_ball_length_Full toss',
                                    'prev_ball_length_Good length', 'prev_ball_length_Short', 'prev_ball_length_Yorker',
                                    'cluster_name_Bowler',
                                    'prev_ball_length_Short of length', 'cluster_name_Aggressive',
                                    'cluster_name_Balanced']

# Define the bin edges
bin_edges = [-1, 0, 6, float('inf')]

# Define the bin labels
bin_labels = [0, 1, 2]


def predict(row):
    return loaded_model.predict(row)


def make_predictions(selected_ids):
    predictions = []

    predict_data = data[(data['id'].isin(selected_ids))]
    processed_data, actual_data = preprocess_data(predict_data)

    for index, row in processed_data.iterrows():
        observation = row.to_numpy().reshape(1, -1)
        prediction = predict(observation)
        actual_ball_length = actual_data.loc[index, 'ball_length']
        predictions.append({ 'actual': actual_ball_length, 'prediction': prediction[0]})

    return predictions


def preprocess_data(row):
    label_encoder = label_encoder_org.copy()
    scaler = scaler_org.copy()

    row['required_rate_bin'] = pd.cut(row['required_rate'], bins=bin_edges, labels=bin_labels)
    row['runs_before_ball'] = row['runs_before_ball'].apply(map_over_range)
    row['probable_length'] = row.apply(
        lambda entry: calculate_probable_length(entry),
        axis=1
    )

    # Fit and transform the probable_length column
    row['probable_length'] = label_encoder.fit_transform(row['probable_length'])

    row_encoded = pd.get_dummies(row[categorical_features])
    #row_scaled = scaler.fit_transform(row[all_numerical_features])
    row_scaled = scaler.transform(row[all_numerical_features].values.reshape(1, -1))

    print("row columns :", row.columns)
    print("row encoded columns :", row_encoded.columns)
    #print("row scled columns :", row_scaled.columns)

    print("row encode shape :", row_encoded.shape)
    print("row scaled shape :", row_scaled.shape)

    # Convert the scaled array to a DataFrame with appropriate column names
    row_scaled_df = pd.DataFrame(row_scaled, columns=all_numerical_features_scaled)

    print("row scaled columns :", row_scaled_df.columns)
    print("row scaled df shape :", row_scaled_df.shape)

    print("original row shape :", row.shape)

    row = row.reset_index(drop=True)
    row_encoded = row_encoded.reset_index(drop=True)
    row_scaled_df = row_scaled_df.reset_index(drop=True)
    print("fsdaf")

    # Concatenate the DataFrames along the columns (axis=1)
    row_processed = pd.concat([row, row_encoded, row_scaled_df], axis=1)

    print("row row_processed shape :", row_processed.shape)
    #row_processed = pd.concat([row, row_encoded, pd.DataFrame(row_scaled, columns=all_numerical_features_scaled)], axis=1)

    # Add missing columns to the encoded user input DataFrame with values set to 0
    missing_columns = set(all_categorical_features_encoded) - set(row_processed.columns)
    for column in missing_columns:
        row_processed[column] = 0

    row_processed = row_processed[selected_features]

    return row_processed, row


def map_over_range(value):
    if value == 0:
        return 0
    elif 0 < value <= 2:
        return 1
    elif 3 <= value <= 5:
        return 2
    else:
        return 3
