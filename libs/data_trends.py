import datetime
import pandas as pd
from pytrends.request import TrendReq


def get_trends(keywords) -> pd.DataFrame:
   #keywords = ['/m/0k8z', '/m/027lnzs', '/m/0mbxw']
   #keywords = ['iPhone', 'Apple',  'iMac']
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=6)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    pytrends = TrendReq()

    pytrends.build_payload(keywords, cat=0, timeframe=f'{start_date_str} {end_date_str}')
    trends_data_whole = pytrends.interest_over_time()
    trends_data_whole = trends_data_whole.iloc[:,:-1]
    return trends_data_whole

def filter_trends(dates, trends):
    result = trends[trends.index.isin(dates)]
    return result