import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

def tech_indicators(prices) -> pd.DataFrame:
    close_price = prices.iloc[:,3]
    m_avg_5 = SMAIndicator(close=close_price, window=5).sma_indicator()
    m_avg_10 = SMAIndicator(close=close_price, window=10).sma_indicator()
    m_avg_20 = SMAIndicator(close=close_price, window=20).sma_indicator()
    ema_5 = EMAIndicator(close=close_price, window=5).ema_indicator()
    ema_10 = EMAIndicator(close=close_price, window=10).ema_indicator()
    ema_20 = EMAIndicator(close=close_price, window=20).ema_indicator()

    bb = BollingerBands(close=close_price, window=20, window_dev=2)
    middle = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()

    temp1 = MACD(prices.iloc[:,0])
    temp2 = MACD(prices.iloc[:,1])
    temp3 = MACD(prices.iloc[:,2])
    temp4 = MACD(close_price)

    MACDLine_High = temp1.macd()
    MACDLine_Low = temp2.macd()
    MACDLine_Open = temp3.macd()
    MACDLine_Close = temp4.macd()

    rsi = RSIIndicator(close=close_price, window=14)
    index = rsi.rsi()

    new_data = pd.DataFrame({
        'MAVG_5': m_avg_5,
        'MAVG_10': m_avg_10,
        'MAVG_20': m_avg_20,
        'EMA_5': ema_5,
        'EMA_10': ema_10,
        'EMA_20': ema_20,
        'Lower_BB': lower,
        'Middle_BB': middle,
        'Upper_BB': upper,
        'RSI': index,
        'MCDA_High': MACDLine_High,
        'MCDA_Low': MACDLine_Low,
        'MCDA_Open': MACDLine_Open,
        'MCDA_Close': MACDLine_Close
    })
    return new_data

def fin_fourier(closing_prices) -> pd.DataFrame:
    n = len(closing_prices)
    fft_result = fft(closing_prices)
    components = [3, 6, 9]
    reconstructed_signals = np.zeros((len(components), n))
    for i, num_components in enumerate(components):
        truncated_fft = np.copy(fft_result)
        truncated_fft[num_components + 1:-num_components] = 0
        reconstructed_signal = ifft(truncated_fft, n=n).real
        reconstructed_signals[i, :] = reconstructed_signal

    fourier_data = pd.DataFrame({
        'Fourier_3': reconstructed_signals[0, :],
        'Fourier_6': reconstructed_signals[1, :],
        'Fourier_9': reconstructed_signals[2, :]
    })
    return fourier_data