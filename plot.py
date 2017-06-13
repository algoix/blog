"""Plot High prices for IBM"""

import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    df = pd.read_csv("data/IBM.csv")
    #Your code here
    """ plot High and CLose two columns"""
    #df[['High','Close']].plot()
    df.High.plot()
    plt.show()  # must be called to show plots


if __name__ == "__main__":
    test_run()
