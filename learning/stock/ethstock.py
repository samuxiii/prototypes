import io
import random
import numpy as np
import pandas as pd
import sklearn
import requests

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

def main():
    data = gettingData()

    print("Retrieved data:")
    print(data.tail())


if __name__ == "__main__":
    main()
