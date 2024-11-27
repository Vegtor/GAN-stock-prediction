import csv
from datetime import timedelta

import pandas as pd
import numpy as np
from data_finance import get_fin_data_mult, check_empty_target, get_fin_data_indiv
from data_indicators_fft import tech_indicators, fin_fourier
from libs.data_trends import get_trends, filter_trends


def get_only(data, file_tickers) -> pd.DataFrame:
    # Načítání názvů ------------------###
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    next(reader)
    used_indicators = next(reader)
    file.close()
    #----------------------------------###
    return data[used_indicators]

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for i in range(0,result.shape[1]):

        for j in range(0,result.shape[0]):
            if np.isnan(result.iloc[j, i]):
                if j > 0 and not np.isnan(result.iloc[j - 1, i]):
                    prev_value = result.iloc[j - 1, i]
                else:
                    prev_value = np.nan

                if j < len(result) - 1:
                    next_value = next(x for x in result.iloc[j+1:,i] if not np.isnan(x))
                    #next_value = result.iloc[j + 1, i]
                else:
                    next_value = np.nan

                if (not np.isnan(prev_value) and not np.isnan(next_value)) and (prev_value != 0 and next_value != 0):
                    result.iloc[j, i] = (prev_value + next_value) / 2
                elif np.isnan(prev_value) and not np.isnan(next_value):
                    result.iloc[j, i] = next_value
                elif not np.isnan(prev_value) and np.isnan(next_value):
                    result.iloc[j, i] = prev_value
                else:
                    result.iloc[j, i] = 0
    return result

def get_whole_data(date_start, date_end, file_tickers, target) -> pd.DataFrame:
    # Načítání názvů ------------------###
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    name_tickers = next(reader)
    file.close()
    # ----------------------------------###
    date_start = date_start - timedelta(days=42)
    whole_data = get_fin_data_mult(name_tickers, date_start, date_end)
    whole_data = check_empty_target(whole_data, target)
    whole_data = get_only(whole_data, file_tickers)
    whole_data = fill_missing(whole_data)

    target_prices = get_fin_data_indiv(target, date_start, date_end)
    target_prices = target_prices.iloc[:, :-2]

    mavg = tech_indicators(target_prices)
    fft = fin_fourier(target_prices.iloc[:, 3])
    mavg = mavg.iloc[-3:, :]
    fft = fft.iloc[-3:, :]

    keywords = ['iPhone', 'Apple', 'iMac']
    trends = get_trends(keywords)
    trends = filter_trends(list(target_prices.index.values), trends)

    whole_data = whole_data.merge(mavg, left_index=True, right_index=True, how='outer')
    whole_data = whole_data.merge(trends, left_index=True, right_index=True, how='outer')
    whole_data = whole_data.merge(fft, left_index=True, right_index=True, how='outer')

    return whole_data[date_start:]

