import csv
import pandas as pd
import numpy as np


def leave_only(data, file_tickers) -> pd.DataFrame:
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

