import os
import io
import random
import numpy as np
import pandas as pd
import sklearn
import requests

'''
Helper function
'''
clear = lambda: os.system('clear')


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

    print("\n\n")


if __name__ == "__main__":
    main()
