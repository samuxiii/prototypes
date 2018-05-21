import io
import random
import numpy as np
import pandas as pd
import sklearn
import requests

def getData():
    url = "https://www.coingecko.com/price_charts/export/279/eur.csv"
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    print(data.head())



def main():
    getData()


if __name__ == "__main__":
    main()
