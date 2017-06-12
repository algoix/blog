#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

import pandas as pd


def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("data/AAPL.csv")
    #Print last 5 rows of the data frame
    print df.tail(5)

if __name__ == "__main__":
    test_run()
