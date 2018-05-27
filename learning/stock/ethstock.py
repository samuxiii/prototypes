import os
import io
import random
import numpy as np
import pandas as pd
import sklearn
import requests
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten


'''
Helper function
'''
clear = lambda: os.system('clear')

#Transform the dataset to shapes defined by 7 steps and 3 features
def prepare_sequence(data):
    sequence = []
    sequence_size = 7
    samples = 0
    for i in range(0, data.shape[0] - sequence_size + 1):
        sequence.append(data[i:i+7])
        samples += 1
    return np.concatenate(sequence).reshape((samples, sequence_size, data.shape[1]))


'''
Data functions
'''
def gettingData():
    url = "https://www.coingecko.com/price_charts/export/279/eur.csv"
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    return data

def preprocessing(data):
    #customize index
    data.snapped_at[0].split()[0]
    data.snapped_at = data.snapped_at.apply(lambda x: x.split()[0])
    data.set_index('snapped_at', inplace=True)
    data.index = pd.to_datetime(data.index)

    '''
    In some cases there is no sample for a certain date.
    '''
    #Generate all the possible days and use them to reindex
    start = data.index[data.index.argmin()]
    end = data.index[data.index.argmax()]
    
    index_complete = pd.date_range(start, end)
    data = data.reindex(index_complete)

    #Fill the blanks with the mean between the previous and the day after
    print("\nLooking if the index is complete...")
    for idx in data.index:
        dayloc = data.index.get_loc(idx)
        day = data.loc[idx]
        if day.hasnans:
            #updating
            rg = slice(dayloc-1, dayloc+2)
            data.loc[idx] = data.iloc[rg].mean()
            print("Day <{}> has been updated with the mean values".format(idx))


    '''
    Adding the target for every sample
    '''
    new_column = 'closed_price'
    datab = data.copy()
    
    nc = list()
    
    for idx in data.index:
        dayloc = data.index.get_loc(idx)
        
        #we put the price in the day after as closed price
        if dayloc == len(data.index)-1:
            #last position will not have closed_price
            closed_price = np.nan
        else:
            closed_price = data.iloc[dayloc+1].price
        
        nc.append(closed_price)
    
    data[new_column] = nc
    #Delete last because we don't know still the closed price 
    data = data.drop(data.index[len(data)-1])

    return data


def get_train_test(data, train_size=0.8):
    split = round(len(data)*train_size)
    data_train, data_test = data[:split].copy(), data[split:].copy()

    return data_train, data_test


def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(7, 3) ))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

'''
Main program
'''
def main():
    clear()

    data = gettingData()
    print("\nRetrieved data:")
    print(data.tail())

    data = preprocessing(data)
    print("\nPreprocessed data:")
    print(data.tail())

    data_train, data_test = get_train_test(data)
    print("\nSplitting the data:")
    print("Size training set: {}".format(data_train.shape[0]))
    print("Size testing set: {}".format(data_test.shape[0]))

    model = build_model()

    #preparing data
    features = ['price', 'market_cap', 'total_volume']
    X_train = prepare_sequence(data_train[features])
    #TODO: recheck this, it's not as simple as that
    y_train = data_train.iloc[-len(X_train):].closed_price.values

    #fit the model
    print("\nTraining...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    #Predicting
    X_test = prepare_sequence(data_test[features])
    pred = model.predict(X_test[-1].reshape(1,7,3))
    print("Prediction tomorrow: {}".format(pred))

    print("\n\n")


if __name__ == "__main__":
    main()
