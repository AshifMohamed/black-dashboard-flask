import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
from apps.data_loader import pattern_transition_matrices

first_innings_transition_matrix = pattern_transition_matrices['first_innings_transition_matrix']
second_innings_transition_matrix = pattern_transition_matrices['second_innings_transition_matrix']
boundary_transition_matrix = pattern_transition_matrices['boundary_transition_matrix']
non_boundary_transition_matrix = pattern_transition_matrices['non_boundary_transition_matrix']
dot_ball_transition_matrix = pattern_transition_matrices['dot_ball_transition_matrix']
min_runs_transition_matrix = pattern_transition_matrices['min_runs_transition_matrix']
max_runs_transition_matrix = pattern_transition_matrices['max_runs_transition_matrix']
bowler_cluster_transition_matrix = pattern_transition_matrices['bowler_cluster_transition_matrix']
aggressive_cluster_transition_matrix = pattern_transition_matrices['aggressive_cluster_transition_matrix']
balanced_cluster_transition_matrix = pattern_transition_matrices['balanced_cluster_transition_matrix']

equivalent_states = [("Full toss", "Yorker"), ("Yorker", "Full toss")]
ball_lengths = ['Full', 'Full toss', 'Good length', 'Short', 'Short of length', 'Yorker']


def calculate_state_change_prob(transition):
    # Calculate total state changes
    total_state_changes = transition.sum()

    # Initialize variables to calculate total probability of state change
    total_state_change_probability = 0.0

    # Calculate total probability of state change to a different states
    for i in range(len(ball_lengths)):
        for j in range(len(ball_lengths)):
            state_change = (ball_lengths[i], ball_lengths[j])

            if ball_lengths[i] != ball_lengths[j] and state_change not in equivalent_states:
                total_state_change_probability += transition[i][j]

    total_probability = round(total_state_change_probability / total_state_changes, 2)
    return round(total_probability * 100, 2)


def calculate_state_change_probabilities():
    state_change_prob_data = []
    state_change_prob_labels = []
    matrices = [
        {
            'label': "Dot Ball",
            'matrix': dot_ball_transition_matrix
        },
        {
            'label': "Min Runs",
            'matrix': min_runs_transition_matrix
        },
        {
            'label': "Max Runs",
            'matrix': max_runs_transition_matrix
        },
    ]
    for entry in matrices:
        prob = calculate_state_change_prob(entry['matrix'].values)
        state_change_prob_data.append(prob)
        state_change_prob_labels.append(entry['label'])

    return {'state_change_prob_data': state_change_prob_data,
            'state_change_prob_labels': state_change_prob_labels}


def calculate_length_change_prob(transition, good_length_states, length_change_states, total_state_changes):
    good_length_changes = 0.0
    # Calculate total state changes
    # total_state_changes = transition.sum()

    # Initialize variables to calculate total probability of state change
    total_length_change_probability = 0.0

    # Calculate total probability of state change to a different state
    for i in range(len(ball_lengths)):
        for j in range(len(ball_lengths)):
            state_change = (ball_lengths[i], ball_lengths[j])
            #             if((ball_lengths[i] == 'Good length') | (ball_lengths[j] == 'Good length')):
            if state_change in good_length_states:
                # print(transition[i][j])
                good_length_changes += transition[i][j]
                continue

            if state_change in length_change_states:
                # print(transition[i][j])
                total_length_change_probability += transition[i][j]

    total_state_changes = total_state_changes - good_length_changes
    # print("Total state ",total_state_changes)
    # print("Total length change ",total_length_change_probability)
    total_probability = total_length_change_probability / total_state_changes
    total_probability = round(total_probability, 2)
    return round(total_probability * 100, 2)


def calculate_length_change_probabilities():
    full_length_change_states = [("Full", "Short"), ("Full toss", "Short"), ("Yorker", "Short"),
                                 ("Full", "Short of length"), ("Full toss", "Short of length"),
                                 ("Yorker", "Short of length")]

    full_to_good_length_states = [("Full", "Good length"), ("Full toss", "Good length"), ("Yorker", "Good length")]

    short_length_change_states = [("Short", "Full"), ("Short", "Full toss"), ("Short", "Yorker"),
                                  ("Short of length", "Full"), ("Short of length", "Full toss"),
                                  ("Short of length", "Yorker")]

    short_to_good_length_states = [("Short", "Good length"), ("Short of length", "Good length")]

    transition = max_runs_transition_matrix.values
    full = calculate_length_change_prob(transition, full_to_good_length_states, full_length_change_states, 3)
    short = calculate_length_change_prob(transition, short_to_good_length_states, short_length_change_states, 2)

    state_change_prob_data = [full, short]
    state_change_prob_labels = ['Full', 'Short']

    return {'state_change_prob_data': state_change_prob_data,
            'state_change_prob_labels': state_change_prob_labels}


def calculate_ball_type_probability(transition_matrix, ball_type):
    # Find the column index corresponding to 'ball_length'
    ball_length_index = np.where(transition_matrix.columns == ball_type)[0][0]

    # Extract the column representing transitions to 'ball_length' from any previous state
    transition_probabilities_to_ball_length = transition_matrix.iloc[:, ball_length_index]

    # Calculate the overall likelihood of transitioning to 'ball_length' from any previous state
    overall_likelihood_to_ball_length = transition_probabilities_to_ball_length.mean()
    return round(overall_likelihood_to_ball_length * 100, 2)


def calculate_ball_type_probabilities():
    labels = ball_lengths.copy()
    labels.remove("Full toss")

    first_inning_short_ball_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Short")
    first_inning_short_of_length_ball_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Short of length")
    first_inning_good_length_ball_length_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Good length")
    first_inning_yorker_ball_length_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Yorker")
    first_inning_full_toss_ball_length_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Full toss")
    first_inning_full_ball_length_prob = calculate_ball_type_probability(first_innings_transition_matrix, "Full")

    second_inning_short_ball_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Short")
    second_inning_short_of_length_ball_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Short of length")
    second_inning_good_length_ball_length_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Good length")
    second_inning_yorker_ball_length_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Yorker")
    second_inning_full_toss_ball_length_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Full toss")
    second_inning_full_ball_length_prob = calculate_ball_type_probability(second_innings_transition_matrix, "Full")

    type_data = [
        [first_inning_full_ball_length_prob, first_inning_good_length_ball_length_prob, first_inning_short_ball_prob,
         first_inning_short_of_length_ball_prob,
         (first_inning_yorker_ball_length_prob + first_inning_full_toss_ball_length_prob)],

        [second_inning_full_ball_length_prob, second_inning_good_length_ball_length_prob,
         second_inning_short_ball_prob,
         second_inning_short_of_length_ball_prob,
         (second_inning_yorker_ball_length_prob + second_inning_full_toss_ball_length_prob)]
        ]
    return {'ball_type_prob_data': type_data,
            'ball_type_prob_labels': labels}


def run_markov_chain(transition_matrix, n=7):
    step = transition_matrix
    for time_step in range(1, n):
        next_step = np.matmul(step, transition_matrix).round(2)

        if np.array_equal(step, next_step):
            return step
        else:
            step = next_step
    return step


def start_to_target(start_length, target_length, n_transitions, transition):
    # Calculate the n-step transition matrix
    transition_matrix_n = run_markov_chain(transition.values, n_transitions)

    # Get the probability of transitioning from start_length to target_length after n_transitions steps
    probability = transition_matrix_n[transition.index.get_loc(start_length), transition.columns.get_loc(target_length)]
    return round(probability * 100, 2)


def get_start_to_target():
    start_length = "Full"
    n_transitions = 3
    probabilities = []

    for ball_length in ball_lengths:
        # Get the probability of transitioning from start_length to target_length after n_transitions steps
        probability = start_to_target(start_length, ball_length, n_transitions, first_innings_transition_matrix)
        probabilities.append(probability)

    return {'start_to_target_data': probabilities,
            'start_to_target_labels': ball_lengths}