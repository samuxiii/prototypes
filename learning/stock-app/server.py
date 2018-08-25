from flask import Flask
app = Flask(__name__)

import pandas as pd
import numpy as np
from stock import Stock, Helper

@app.route("/")
def hello():
    return "Hello, man!"

@app.route("/stock")
def stock():
    return predictStock(Stock(), Helper())


'''
Main program
'''
def predictStock(stock, helper):

    data = stock.gettingData()

    #Store the last date. Useful for print outs
    prediction_date = pd.to_datetime(data.iloc[-1].snapped_at).strftime("%Y-%m-%d")
    data = stock.preprocessing(data)

    data_train, data_test = stock.get_train_test(data)
    data_train, data_test, scaler = stock.scale(data_train, data_test)

    #preparing data
    features = ['price', 'market_cap', 'total_volume']
    X_train = helper.prepare_sequence(data_train[features])

    #customize y_train for sequence
    y_train = data_train.iloc[6:].closed_price.values

    #fit the model
    model = stock.build_model()
    model = stock.training(model, X_train, y_train)

    #Predicting
    X_test = helper.prepare_sequence(data_test[features])
    last_sequence = X_test[-1].reshape(1,7,3)

    pred = model.predict(last_sequence).item((0,0))
    #recover real value (not normalized)
    pred = scaler.inverse_transform(np.array([[pred,0,0,0]]))

    result = "Prediction for {}: {:.2f}".format(prediction_date, pred.item((0,0)))
    print(result)
    return result

