from apps.data_loader import transition_matrices, weight_innings_id, weight_prev_isBoundary


def calculate_probable_length(row):
    boundary_transition = []
    innings_transition = []

    if row['prev_isBoundary'] == 1:
        boundary_transition = transition_matrices['boundary_transition_matrix']
    else:
        boundary_transition = transition_matrices['non_boundary_transition_matrix']

    if row['innings_id'] == 1:
        innings_transition = transition_matrices['first_innings_transition_matrix']
    else:
        innings_transition = transition_matrices['second_innings_transition_matrix']

    # Calculate highest probabilities and corresponding ball lengths
    if row['prev_ball_length'] in innings_transition.index:
        max_prob_innings_id = innings_transition.loc[row['prev_ball_length']]
        prob_innings_id = max_prob_innings_id.max()
        ball_length_innings_id = max_prob_innings_id.idxmax()
    else:
        prob_innings_id = 0
        ball_length_innings_id = ""

    if row['prev_ball_length'] in boundary_transition.index:
        max_prob_prev_isBoundary = boundary_transition.loc[row['prev_ball_length']]
        prob_prev_isBoundary = max_prob_prev_isBoundary.max()
        ball_length_prev_isBoundary = max_prob_prev_isBoundary.idxmax()
    else:
        prob_prev_isBoundary = 0
        ball_length_prev_isBoundary = ""

    if prob_innings_id == 0 and prob_prev_isBoundary == 0:
        return transition_matrices['overall_transition_matrix'].loc[row['prev_ball_length']].idxmax()

    if ball_length_innings_id == ball_length_prev_isBoundary:
        return ball_length_innings_id

    # Calculate weighted probabilities
    weighted_prob_innings_id = prob_innings_id * weight_innings_id
    weighted_prob_prev_isBoundary = prob_prev_isBoundary * weight_prev_isBoundary

    # Choose the most probable ball length
    if weighted_prob_innings_id > weighted_prob_prev_isBoundary:
        return ball_length_innings_id
    else:
        return ball_length_prev_isBoundary
