
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt

from dateutil.parser import parse


################################################################################
# Recieve data
################################################################################

DATA_FOLDER = "data"


def download_data(symbol, start, end):
	"""Returns an request object containing the data for the specified symbol"""
	s = parse(start)
	e = parse(end)

	url = "http://chart.finance.yahoo.com/table.csv"
	payload = {'s' : str(symbol), 'a' : s.month-1, 'b' : s.day, 'c' : s.year,
		'd' : e.month-1, 'e' : e.day, 'f' : e.year, 'g' : 'd', 'ignore' : '.csv'}

	return requests.get(url, params=payload)


def get_csv(symbols, start='1997-01-01', end='2016-09-01'):
	"""Save the stock data for the given list on symbols""" 
	for symbol in symbols:
		if not symbol_is_file(symbol): 
			request = download_data(symbol, start, end)
			path = symbol_to_path(symbol)
			f = open(path, 'w')
			f.write(request.text)
			f.close()

def symbol_is_file(symbol, base_dir=DATA_FOLDER):
	"""Return True if CSV file exists or False if not"""
	return os.path.isfile(base_dir + "/{}.csv".format(str(symbol)))

def symbol_to_path(symbol, base_dir=DATA_FOLDER):
	"""Return CSV file path given ticker symbol."""
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	get_csv(symbols)
	df = pd.DataFrame(index=dates)
	if 'SPY' not in symbols: #add SPY for reference, if absent
		symbols.insert(0,'SPY')

	for symbol in symbols:
		get_csv(symbol)
		df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
			parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
		df_temp = df_temp.rename(columns={'Adj Close': symbol})
		df = df.join(df_temp)
		if symbol == 'SPY': #drop dates SPY did not trade
			df = df.dropna(subset=["SPY"])

	return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
	"""Plot stock prices with a custom title and meaningful axis labels."""
	ax = df.plot(title=title, fontsize=12)
	ax.set_xlabel(str(xlabel))
	ax.set_ylabel(str(ylabel))
	plt.show()


################################################################################
# Data Manipulation
################################################################################

def fill_missing_values(df):
    """Fill missing values in data frame, in place."""
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

def get_daily_returns(df):
	"""Compute and return the daily return values."""
	daily_returns = df.copy()
	daily_returns[1:] = (df[1:] / df[:-1].values) - 1
	daily_returns.iloc[0] = 0 # set daily returns for row 0 to 0
	return daily_returns

def get_rolling_mean(values, window=20):
	"""Return rolling mean of given values, using specified window size."""
	return values.rolling(window, center=False).mean()

def get_rolling_std(values, window=20):
	"""Return rolling standard deviation of given values, using specified window size"""
	return values.rolling(window, center=False).std()

def get_bollinger_bands(rm, rstd):
	"""Return upper and lower Bollinger Bands."""
	upper_band = rm + 2*rstd
	lower_band = rm - 2*rstd
	return upper_band, lower_band