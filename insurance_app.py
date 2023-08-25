from flask import Flask,request,render_template
import pickle
import numpy as np
import sklearn


# importing model

model = pickle.load(open('insurance.pkl1','rb'))


# creating flask app

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods = ['POST'])
def predict():
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    sex = int(request.form['sex'])
    region = int(request.form['region'])

    feature_list = [age,bmi,children,smoker,sex,region]
    single_pred = np.array(feature_list).reshape(1,-1)

    prediction = model.predict(single_pred)

    return render_template('index.html',result = prediction)
# python name

if __name__ == '__main__':
    app.run(debug=True)

