from flask import Flask

# import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file
import pandas as pd
from prediction_of_UserInput.prediction_file import Prediction_from_api

from file_operation import file_op
import numpy as np
import pandas as pd
import json
import os

application = Flask(__name__)


@application.route("/", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        department = request.form['department']
        region = request.form['region']
        education = request.form['education']
        gender = request.form['gender']
        Recruitment_Channel = request.form['Recruitment_Channel']
        no_of_trainings = int(request.form['no_of_trainings'])
        age = int(request.form['age'])
        Previous_Year_Rating = float(request.form['Previous_Year_Rating'])
        length_of_service = int(request.form['length_of_service'])
        KPIs_met = int(request.form['KPIs_met >80%'])
        awards_won = int(request.form['awards_won?'])
        avg_training_score = int(request.form['avg_training_score'])
        print(department, region, education,gender,Recruitment_Channel, no_of_trainings, age, Previous_Year_Rating,length_of_service,KPIs_met,awards_won,avg_training_score)

        pred = Prediction_from_api(department, region, education,gender,Recruitment_Channel, no_of_trainings, age, Previous_Year_Rating,length_of_service,KPIs_met,awards_won,avg_training_score)
        pred_data = pred.prediction_api()

        return render_template('form.html', pred_data_msg=f'Hello {name} {pred_data}')
    return render_template('form.html')


if __name__ == '__main__':
    application.run(debug=True)
