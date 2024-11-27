import pandas as pd
import csv
import yfinance as yf
from datetime import datetime, timedelta
from libs.data_edit import fill_missing
#import numpy as np
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter


def check_empty_target(df, target) -> pd.DataFrame:
    last_row = df.iloc[-1]
    columns_with_target = [col for col in df.columns if target in col]

    if any(pd.isna(last_row[col]) for col in columns_with_target):
        df = df.head(-1)
    return df

def get_fin_data_3_day(file_tickers, target) -> pd.DataFrame:
    # Načítání názvů ------------------###
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    name_tickers = next(reader)
    file.close()
    #----------------------------------###

    # Časový interval ----------------------------------------###
    if datetime.today().weekday() < 5:
        shift = 2
    elif datetime.today().weekday() == 5:
        shift = 3
    else:
        shift = 4
    date_start = datetime.now().date() - timedelta(days=shift)
    date_end = datetime.now().date() + timedelta(days=1)
    #---------------------------------------------------------###

    whole_data = get_fin_data_mult(name_tickers, date_start, date_end)
    whole_data = check_empty_target(whole_data, target)

    if whole_data.shape[0] > 3:
        nan_count = whole_data.iloc[-1,:].isna().sum()
        if nan_count > 0.10*whole_data.shape[1]:
            whole_data = whole_data.head(-1)
        else:
            whole_data = whole_data.tail(-1)
    whole_data = fill_missing(whole_data)
    return whole_data
#whole_data.to_csv("data/test.csv")

def get_fin_data_mult(name_tickers, date_start, date_end) -> pd.DataFrame:
    # Session nastavení a omezení volání na 2 za 5 sekund ----------------####
    class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
        pass
    session = CachedLimiterSession(limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
                                   bucket_class=MemoryQueueBucket,
                                   backend=SQLiteCache("yfinance.cache"))
    #----------------------------------------------------------------------###
    ticker_sets = yf.Tickers(name_tickers, session=session)

    whole_data = pd.DataFrame()
    for i in range(len(name_tickers)):
        hist_price = ticker_sets.tickers[name_tickers[i]].history(start=date_start, end=date_end, interval='1d',
                                                                  auto_adjust=False, actions=False)
        hist_price = hist_price.rename(columns={"Open": name_tickers[i] + "_Open", "High": name_tickers[i] + "_High",
                                                "Low": name_tickers[i] + "_Low", "Close": name_tickers[i] + "_Close",
                                                "Adj Close": name_tickers[i] + "_Adj_Close",
                                                "Volume": name_tickers[i] + "_Volume"})

        hist_price.index = hist_price.index.date
        hist_price.index.name = "Date"
        if whole_data.empty:
            whole_data = hist_price
        else:
            whole_data = whole_data.merge(hist_price, left_index=True, right_index=True, how='outer')
    return whole_data

def get_fin_data_indiv(name_ticker, date_start, date_end) -> pd.DataFrame:
    # Session nastavení a omezení volání na 2 za 5 sekund ----------------####
    class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
        pass
    session = CachedLimiterSession(limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
                                   bucket_class=MemoryQueueBucket,
                                   backend=SQLiteCache("yfinance.cache"))
    #----------------------------------------------------------------------###
    ticker_sets = yf.Ticker(name_ticker, session=session)

    hist_price = ticker_sets.history(start=date_start, end=date_end, interval='1d',
                                                                  auto_adjust=False, actions=False)
    hist_price = hist_price.rename(columns={"Open": name_ticker + "_Open", "High": name_ticker + "_High",
                                                "Low": name_ticker + "_Low", "Close": name_ticker + "_Close",
                                                "Adj Close": name_ticker + "_Adj_Close",
                                                "Volume": name_ticker + "_Volume"})

    hist_price.index = hist_price.index.date
    hist_price.index.name = "Date"
    whole_data = hist_price
    return whole_data