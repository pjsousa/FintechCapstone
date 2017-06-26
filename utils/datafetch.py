import pandas_datareader.data as web
import datetime
import pandas as pd
import io
import requests
import sys

CONFIG_PATH = "./config"
DATA_PATH = "./data"
RAW_DATA_PATH = "{}/A_RAW".format(DATA_PATH)
BASELINE_DATA_PATH = "{}/C_BASELINE".format(DATA_PATH)
TIER1_DATA_PATH = "{}/B_TIER1".format(DATA_PATH)
TIER2_DATA_PATH = "{}/C_TIER2".format(DATA_PATH)


_EXCHANGES = [
	#"nasdaq"
	"nasdaq100"
	#,"nyse"
	#,"amex"
]

_URLS = dict()
_URLS["nasdaq"] = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
_URLS["nasdaq100"] = "http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download" # Nasdaq100
_URLS["nyse"] = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
_URLS["amex"] = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download"

def print_progress(str_output):
	sys.stdout.write('\r')
	# the exact output you're looking for:
	sys.stdout.write(str_output)
	sys.stdout.flush()

def fetch_quotes(ticker, from_date=datetime.datetime(2005, 1, 1), to_date=datetime.datetime.now(), source="google"):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		_r = web.DataReader(ticker, source, from_date, to_date)
	except:
		_r = None
	return _r


def download_companieslist(exchange):
	url=_URLS[exchange]
	s=requests.get(url).content
	c=pd.read_csv(io.StringIO(s.decode('utf-8')))
	c.to_csv("{}/tickers_{}.csv".format(CONFIG_PATH,exchange), index=False)


def load_exchangesinfos(config_path=CONFIG_PATH, verbose=True, exchange=_EXCHANGES):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	_rr = []
	_distinct_count = None
	_cols_to_use = ["Symbol", "Name", "LastSale", "MarketCap", "IPOyear", "Sector", "industry", "Summary Quote"]

	
	## lets load all info files and append them with the exchange name
	for i, itr in enumerate(_EXCHANGES):
		itr_path = "{}/tickers_{}.csv".format(config_path,itr)
		try:
			itr_df = pd.read_csv(itr_path, usecols=_cols_to_use)
			itr_df["exchange"] = itr
		except:
			download_companieslist(itr)
			itr_df = pd.read_csv(itr_path, usecols=_cols_to_use)
			itr_df["exchange"] = itr

		if verbose:
			print("{} has {} tickers.".format(itr, itr_df.shape[0]))

		_rr.append(itr_df)

	_r = pd.concat(_rr)

	if verbose:
		print("\n")
		print("Final dataset has {} records".format(_r.shape[0]))
		_distinct_count = len(pd.unique(_r["Symbol"]))
		if _r.shape[0] == _distinct_count:
			print("(OK) Final dataset has {} distinct tickers".format(_r.shape[0]))
		else:
			print("(NOK) Final dataset has duplicate tickers ({} distinct)".format(_distinct_count))

	return _r


def duplicate_tickers(df, ign_exchange=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	dups = df[df.duplicated(["Symbol"], keep=False) == True]
	_r = pd.DataFrame(dups)

	if ign_exchange:
		_r["ignoring_exchange"] = _r.duplicated(["Symbol", "Name", "LastSale", "MarketCap", "IPOyear", "Sector", "industry", "Summary Quote"])


	return _r


def initial_dataload(ticker_list, verbose=True, del_temp=False, status_df=None):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	itr_df = None
	itr_err = None
	err_count = 0
	filepath = "{}/{}.csv"
	_len = len(ticker_list)
	_start = datetime.datetime.now()

	print("{} - Fetching {} tickers".format(_start, len(ticker_list)))

	for idx, itr_tkr in enumerate(ticker_list):
		_start = datetime.datetime.now()
		if os.path.isfile(filepath.format(DATA_PATH, itr_tkr)):
			err_count += 1
		else:
			itr_df = fetch_quotes(itr_tkr)
			if itr_df is None:
				itr_err = True
				err_count += 1
			else:
				itr_err = False
				itr_df.to_csv(filepath.format(DATA_PATH, itr_tkr), encoding="utf-8")
				if del_temp:
					del itr_df

		if verbose:
			print_progress("  Fetching {} - ({}/{}) [{} skipped]".format(itr_tkr, idx + 1, _len, err_count))

		if not(status_df is None):
			status_df.loc[itr_tkr, "status"] = "NOK" if itr_err else "OK"
			status_df.loc[itr_tkr, "start"] = _start
			status_df.loc[itr_tkr, "end"] = datetime.datetime.now()

	print("\n{} - Fetching Finished".format(datetime.datetime.now()))

	return None


def load_raw_frame(ticker, tryfetch=True, parseDate=True, dropAdjClose=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		_r = pd.read_csv("{}/{}.csv".format(DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

		if dropAdjClose:
			close_ratio = _r["Close"] / _r["Adj Close"]
			_r["Open"] = _r["Open"] / close_ratio
			_r["High"] = _r["High"] / close_ratio
			_r["Low"] = _r["Low"] / close_ratio
			_r["Close"] = _r["Adj Close"]

			_r.drop("Adj Close", axis=1, inplace=True)
	except Exception as e:
		if tryfetch:
			initial_dataload([ticker], False, True)
			_r = load_raw_frame(ticker, False)
		else:
			_r = None

	return _r


def store_tier1_frame(tier1_df, ticker):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		tier1_df.to_hdf("{}/{}.h5".format(TIER1_DATA_PATH, ticker), "TIER1")
		_r = True
	except:
		_r = False

	return _r


def load_tier1_frame(ticker):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		_r = pd.read_hdf("{}/{}.h5".format(TIER1_DATA_PATH, ticker), "TIER1")
	except:
		_r = None

	return _r



def store_tier2_frame(tier1_df, ticker):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		tier1_df.to_csv("{}/{}.csv".format(TIER2_DATA_PATH, ticker), index=False)
		_r = True
	except:
		_r = False

	return _r


def load_tier2_frame(ticker, parseDate=True):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		_r = pd.read_csv("{}/{}.csv".format(TIER2_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)
	except:
		_r = None

	return _r

def store_baseline_frame(tier1_df, ticker, baselinepath=BASELINE_DATA_PATH):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		tier1_df.to_csv("{}/{}.csv".format(baselinepath, ticker), index=False)
		_r = True
	except:
		_r = False

	return _r


def load_baseline_frame(ticker, baselinepath=BASELINE_DATA_PATH, parseDate=True):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	try:
		_r = pd.read_csv("{}/{}.csv".format(baselinepath, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)
	except:
		_r = None

	return _r





