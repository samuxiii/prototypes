import os
import io
import math
import random
import numpy as np
import pandas as pd
import sklearn
import requests
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.models import load_model


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

#Download files
def download_file(url, filename):
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    total_kb_size = math.ceil(total_size//block_size)
    
    wrote = 0 
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=total_kb_size , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)

'''
Data functions
'''
def gettingData():
    url = "https://www.coingecko.com/price_charts/export/279/eur.csv"
    datafile = 'eth-eur.csv'
    download_file(url, datafile)
    return pd.read_csv(datafile)

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

def scale(data, scaler):

    scaler = StandardScaler()
    data_train_norm, data_test_norm = data_train.copy(), data_test.copy()

    data_train_norm[data.columns] = scaler.fit_transform(data_train[data.columns])
    data_test_norm[data.columns] = scaler.transform(data_test[data.columns])

    return data_train_norm, data_test_norm



def get_train_test(data, train_size=0.9):
    split = round(len(data)*train_size)
    data_train, data_test = data[:split].copy(), data[split:].copy()

    return data_train, data_test


def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(7, 3) ))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def training(model, X, y):
    modelfile = 'model.h5'

    if (os.path.exists(modelfile)):
        print("Recovering backed up model..\n")
        return load_model(modelfile)
    else:
        print("Training...\n")
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        model.save(modelfile)
        return model

'''
Main program
'''
def main():
    clear()

    data = gettingData()
    print("\nRetrieved data:")
    print(data.tail())

    #Store the last date. Useful for print outs
    prediction_date = pd.to_datetime(data.iloc[-1].snapped_at).strftime("%Y-%m-%d")

    data = preprocessing(data)
    print("\nPreprocessed data:")
    print(data.tail())

    data_train, data_test = get_train_test(data)
    #preparing data
    features = ['price', 'market_cap', 'total_volume']
    X_train = prepare_sequence(data_train[features])
    #customize y_train for sequence
    y_train = data_train.iloc[6:].closed_price.values

    print("Size training set: {}".format(X_train.shape[0]))

    #fit the model
    model = build_model()
    model = training(model, X_train, y_train)

    #Predicting
    X_test = prepare_sequence(data_test[features])
    last_sequence = X_test[-1].reshape(1,7,3)
    print("Size testing set: {}".format(X_test.shape[0]))
    print("Last sequence:\n{}".format(last_sequence))

    pred = model.predict(last_sequence)
    print("\nPrediction for {}: {:.2f}".format(prediction_date, pred[0][0]))

    print("\n\n")


if __name__ == "__main__":
    main()
