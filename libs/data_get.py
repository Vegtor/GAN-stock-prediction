import csv
from datetime import timedelta, datetime
import pandas as pd

from libs.data_edit import leave_only, fill_missing
from libs.data_finance import get_fin_data_mult, check_empty_target, get_fin_data_indiv
from libs.data_indicators_fft import tech_indicators, fin_fourier
from libs.data_trends import get_trends_3_days, filter_trends, get_trends


def obtain_whole_data(date_start, date_end, file_tickers, target) -> pd.DataFrame:
    # Načítání názvů ------------------###
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    name_tickers = next(reader)
    file.close()
    # ----------------------------------###
    whole_data = get_fin_data_mult(name_tickers, date_start, date_end)
    whole_data = check_empty_target(whole_data, target)
    whole_data = leave_only(whole_data, file_tickers)
    whole_data = fill_missing(whole_data)

    target_prices = get_fin_data_indiv(target, date_start - timedelta(days=42), date_end)
    target_prices = target_prices.iloc[:, :-2]

    fft = fin_fourier(target_prices.iloc[:, 3])
    fft.index = target_prices.index
    fft = fft[date_start:date_end]

    mavg = tech_indicators(target_prices)
    mavg = mavg[date_start:date_end]
    mavg.index = whole_data.index

    keywords = ['iPhone', 'Apple', 'iMac']
    trends = get_trends(keywords, date_start, date_end)
    trends = filter_trends(list(target_prices.index.values), trends)
    trends.index = whole_data.index

    whole_data = whole_data.merge(mavg, left_index=True, right_index=True, how='outer')
    whole_data = whole_data.merge(trends, left_index=True, right_index=True, how='outer')
    whole_data = whole_data.merge(fft, left_index=True, right_index=True, how='outer')

    return whole_data

def obtain_dataset_3_days(file_tickers, target) -> pd.DataFrame:
    # Načítání názvů ------------------###
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    name_tickers = next(reader)
    file.close()
    # ----------------------------------###
    # Časový interval ----------------------------------------###
    if datetime.today().weekday() < 3:
        shift = 5
    elif datetime.today().weekday() < 5:
        shift = 3
    else:
        shift = 4
    date_start = datetime.now().date() - timedelta(days=shift)
    date_end = datetime.now().date() + timedelta(days=1)
    # ---------------------------------------------------------###

    whole_data = obtain_whole_data(date_start, date_end, file_tickers, target)

    if whole_data.shape[0] > 3:
        nan_count = whole_data.iloc[-1,:].isna().sum()
        if nan_count > 0.10*whole_data.shape[1]:
            whole_data = whole_data.head(-1)
        else:
            whole_data = whole_data.tail(-1)
    return whole_data

