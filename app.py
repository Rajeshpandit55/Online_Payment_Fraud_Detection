from keras.models import load_model
import numpy as np
import pickle as pkl
import sklearn

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("this is the version of Sikit learn:  ", sklearn.__version__)

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = pkl.load(open("model.pkl", "rb"))

print("this is the type of model: ", type(model))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    type = request.form["type"]
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])

    if type == "CASH_OUT":
        val = 1
    elif type == "PAYMENT":
        val = 2
    elif type == "CASH_IN":
        val = 3
    elif type == "CASH_IN":
        val = 4
    else:
        val = 5

    input_array = np.array([[val, amount, oldbalanceOrg, newbalanceOrig]])

    prediction = model.predict(input_array)

    output = prediction[0]

    return render_template("index.html", prediction=output)


if __name__ == "__main__":
    app.run(debug=True)


# https://www.kaggle.com/datasets/kartik2112/fraud-detection

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
