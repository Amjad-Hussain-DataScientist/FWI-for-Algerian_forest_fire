import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app = application #app = Flask(__name__): This line creates an instance of the Flask class. This instance, named app, becomes your entire web application object. The special Python variable __name__ is passed as an argument. This helps Flask determine the root path of the application so it can find resources like templates and static files correctly.

#here our flask will inderact to our pickle file  so import ridge and standart scalar pickle
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

#homepage
@app.route("/")
def index():
    return render_template('index.html')
#to predict data we make another route and method be in the route ['GET','POST']
@app.route("/predictdata",methods=['GET','POST']) #get request mean getting something and post means sending something to server
def predict_datapoint():
    if request.method=='POST':
        # 1. Get the data submitted from the HTML form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region =float(request.form.get('Region'))
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")