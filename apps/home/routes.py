# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, jsonify
from jinja2 import TemplateNotFound
from apps.data_loader import data
from apps.home.predictor import make_predictions
from apps.home.dashboard import calculate_state_change_probabilities, calculate_length_change_probabilities, \
    calculate_ball_type_probabilities, get_start_to_target


# @blueprint.route('/index')
@blueprint.route('/')
def index():
    return render_template('home/dashboard.html', segment='dashboard')


@blueprint.route('/<template>')
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        # filter_data = data.drop("id", axis=1)
        return render_template("home/" + template, segment=segment, data=list(data.values.tolist()))

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


@blueprint.route('/predict', methods=['POST'])
def predict():
    data_id = request.form.get('data_id')
    prediction = make_predictions([data_id])
    return jsonify({'prediction': prediction})


@blueprint.route('/dashboard', methods=['GET'])
def dashboard():
    state_change_data = calculate_state_change_probabilities()
    length_change_data = calculate_length_change_probabilities()
    ball_type_probabilities = calculate_ball_type_probabilities()
    start_to_target = get_start_to_target()
    return jsonify({'state_change_data': state_change_data, 'length_change_data': length_change_data,
                    'ball_type_probabilities': ball_type_probabilities, 'start_to_target': start_to_target})


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
