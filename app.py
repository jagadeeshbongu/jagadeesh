import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__) #Initialize the flask App

model = joblib.load('model.pkl')
poly = joblib.load('poly.pkl')
ohe = joblib.load('enc.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_features = [[x for x in request.form.values()]]

    x_cols = ['Mileage','Kilometers_Driven','Engine','Power','Fuel_Type',
              'Owner_Type','Transmission','Seats','Year']

    inputdata = pd.DataFrame(inp_features, columns = x_cols)

    cats = pd.DataFrame(ohe.transform(inputdata.iloc[:,[4,6]]))
    cats.columns = ohe.get_feature_names()

    inputdata.iloc[:,[0,1,2,3,5,7,8]] = inputdata.iloc[:,[0,1,2,3,5,7,8]].astype(float)

    record = pd.concat([inputdata.iloc[:,[0,1,2,3,5,7,8]],cats], axis=1)
    record1 = poly.transform(record)
    prediction = model.predict(record1)
    output = round(prediction[0], 0)

    return render_template('index.html', Text = "For The Given Properties: {}".format(inp_features[0]),Estimation = "Estimated Price: {} Rupees".format(output))

if __name__ == "__main__":
    app.run(debug=True)
