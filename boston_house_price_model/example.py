from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import pandas as pd

import ValuationTool

from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm


def return_prediction(model, sample_json):
    nr_rooms = sample_json["no_rooms"]
    ptratio = sample_json["pt_ratio"]
    chas = sample_json["charles"]
    conf_interval = sample_json["confidence"]


    doll_est, upper_bound, lower_bound = model.get_dollar_estimate(nr_rooms,ptratio,chas,conf_interval)

    return doll_est


app = Flask(__name__)
app.config['SECRET_KEY']='mysecretkey' #allowas the form to work, makes sure it is not being hacked


class FlowerForm(FlaskForm):

    num_rooms = TextField("number of rooms ")
    ptratio = TextField("student to teacher ratio")
    chas = TextField("Next to Charles River True/False")
    high_confidence = TextField("95% confidence interval if True")

    submit = SubmitField("Predict")



@app.route('/', methods=['GET','POST'])

def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['num_rooms'] = form.num_rooms.data
        session['ptratio'] = form.ptratio.data
        session['chas'] = form.chas.data
        session['high_confidence'] = form.high_confidence.data

        return redirect(url_for("prediction"))
        #only redirect to prediction if the form is validared upon submission
    return render_template('home.html', form=form)


home_model = joblib.load('boston_valuation.pkl')


@app.route('/prediction')
def prediction():
    content = {}
    content['no_rooms'] = float(session['num_rooms'])
    content['pt_ratio'] = float(session['ptratio'])
    content['charles'] = float(session['chas'])
    content['confidence'] = float(session['high_confidence'])

    results = return_prediction(ValuationTool, content)

    return render_template('prediction.html', results=results)


if __name__ =="__main__":
    app.run()





