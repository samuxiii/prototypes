import os
import io
import math
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import requests
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.models import load_model


'''
class Helper
'''
class Helper:

    #Transform the dataset to shapes defined by 7 steps and 3 features
    def prepare_sequence(self, data):
        sequence = []
        sequence_size = 7
        samples = 0
        for i in range(0, data.shape[0] - sequence_size + 1):
            sequence.append(data[i:i+7])
            samples += 1
        return np.concatenate(sequence).reshape((samples, sequence_size, data.shape[1]))
    
    #Download files
    def download_file(self, url, filename):
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
Class Stock
'''
class Stock:

    def __init__(self):
        self.helper = Helper()
        pass

    def gettingData(self):
        url = "https://www.coingecko.com/price_charts/export/279/eur.csv"
        datafile = 'eth-eur.csv'
        self.helper.download_file(url, datafile)
        return pd.read_csv(datafile)
    
    def preprocessing(self, data):
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
    
    def scale(self, data_train, data_test):
    
        scaler = StandardScaler()
        data_train_norm, data_test_norm = data_train.copy(), data_test.copy()
        columns = data_train.columns
    
        data_train_norm[columns] = scaler.fit_transform(data_train[columns])
        data_test_norm[columns] = scaler.transform(data_test[columns])
    
        return data_train_norm, data_test_norm, scaler
    
    
    
    def get_train_test(self, data, train_size=0.9):
        split = round(len(data)*train_size)
        data_train, data_test = data[:split].copy(), data[split:].copy()
    
        return data_train, data_test
    
    
    def build_model(self, ):
        model = Sequential()
        model.add(LSTM(32, input_shape=(7, 3) ))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
    
        return model
    
    def training(self, model, X, y):
        modelfile = 'model.h5'
    
        if (os.path.exists(modelfile)):
            print("Recovering backed up model..\n")
            return load_model(modelfile)
        else:
            print("Training...\n")
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            model.save(modelfile)
            return model


