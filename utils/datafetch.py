import pandas_datareader.data as web
import datetime
import pandas as pd

CONFIG_PATH = "./config"
DATA_PATH = "./data"
RAW_DATA_PATH = "{}/A_RAW".format(DATA_PATH)
TIER1_DATA_PATH = "{}/B_TIER1".format(DATA_PATH)
TIER2_DATA_PATH = "{}/C_TIER2".format(DATA_PATH)

def fetch_quotes(ticker, from_date=datetime.datetime(1900, 1, 1), to_date=datetime.datetime.now(), source="yahoo"):
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

_EXCHANGES = [
	"amex"
	,"nasdaq"
	,"nyse"
]


def load_exchangesinfos(config_path=CONFIG_PATH, verbose=True):
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
	_cols_to_use = ["Symbol", "Name", "MarketCap", "Sector", "Industry"]

	
	## lets load all info files and append them with the exchange name
	for i, itr in enumerate(_EXCHANGES):
		itr_path = "{}/tickers_{}.csv".format(config_path,itr)
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
		_r["ignoring_exchange"] = _r.duplicated(["Symbol", "Name", "MarketCap", "Sector", "Industry"])

	return _r


def initial_dataload(ticker_list, verbose=True, del_temp=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = dict()
	_r["OK"] = []
	_r["NOK"] = []
	itr_df = None
	itr_err = None
	filepath = "{}/{}.csv"
	_len = len(ticker_list)
	_start = datetime.datetime.now()
	_end = None

	for idx, itr_tkr in enumerate(ticker_list):
		itr_df = fetch_quotes(itr_tkr)
		if itr_df is None:
			_r["NOK"].append(itr_tkr)
			itr_err = True
		else:
			itr_err = False
			_r["OK"].append(itr_tkr)
			itr_df.to_csv(filepath.format(DATA_PATH, itr_tkr), encoding="utf-8")
			if del_temp:
				del itr_df

		if verbose:
			if itr_err:
				print("({}/{}) ERROR receiving {}".format(idx + 1, _len, itr_tkr))
			else:
				print("({}/{}) Recv. and Stored {}".format(idx + 1, _len, itr_tkr))
		_end = datetime.datetime.now()
	
	if verbose:
		print("Took {}".format(str(_end - _start)))

	return _r


def load_raw_frame(ticker, tryfetch=True, parseDate=True):
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
	except FileNotFoundError:
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
		tier1_df.to_csv("{}/{}.csv".format(TIER2_DATA_PATH, ticker))
		_r = True
	except:
		_r = False

	return _r


def load_tier2_frame(ticker):
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
	except:
		_r = None

	return _r


