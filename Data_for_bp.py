__author__ = "samanvayms"

# modified from https://github.com/cuemacro/findatapy/blob/master/findatapy_examples/dukascopy_example.py 
# Author for original - saeedamen

from findatapy.market import Market, MarketDataRequest, MarketDataGenerator
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import mplfinance as mpf 
import pickle

def get_tick_data(start_date,finish_date):
    market = Market(market_data_generator=MarketDataGenerator())

    md_request = MarketDataRequest( start_date=start_date,
                                finish_date=finish_date,
                                fields=['bid', 'ask'], 
                                vendor_fields=['bid', 'ask'],
                                freq='tick', 
                                data_source='dukascopy',
                                tickers=['EURUSD'],
                                vendor_tickers=['EURUSD'])
    df = market.fetch_market(md_request)
    df['EURUSD.mid'] = (df['EURUSD.ask'] + df['EURUSD.bid']) / 2.0
    
    return df

def tick_to_ohlc(tick_df: pd.DataFrame, timeframe: str,pickle_file_name_ohlc) -> pd.DataFrame:
    '''
    Convert tick data to OHLC data.
    1 min = 1T
    1 Hr = 1H
    1 Day = 1D
    '''
    # Assuming the DataFrame is indexed by timestamp and has a 'Mid' column
    # for the mid prices. Adapt as necessary.

    # Resample the tick data to OHLC data using the specified timeframe
    ohlc_df = tick_df['EURUSD.mid'].resample(timeframe).ohlc()
    
    # Drop rows where all values are NaN (which may happen in less active trading periods)
    ohlc_df.dropna(how='all', inplace=True)
    
    # write to pickle file
    ohlc_df.to_pickle(pickle_file_name_ohlc)
    return ohlc_df

def get_candlestick_data(start_date,finish_date,timeframe: str):
    '''get candlestick data from start_date to finish_date with timeframe
    date format: 'dd mmm yyyy' like '14 Jun 2016'
    timeframe format:
    1 min = 1T
    1 Hr = 1H
    1 Day = 1D
    '''
    tick_data = get_tick_data(start_date,finish_date)
    ohlc_df = tick_to_ohlc(tick_data, timeframe)
    return ohlc_df,tick_data

def get_date_pairs(start_date_str,end_date_str,date_format='%d %b %Y'):
    # Parse the start and end dates
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)

    # Initialize an empty list to hold the interval pairs
    interval_list = []

    # Increment by one month until we reach or surpass the end date
    current_date = start_date
    next_date = current_date + relativedelta(months=1)
    while next_date <= end_date:
        interval_list.append((current_date.strftime(date_format), next_date.strftime(date_format)))
        current_date = next_date
        next_date += relativedelta(months=1)

    # Add the remaining interval if there are extra days left
    if current_date != end_date:
        interval_list.append((current_date.strftime(date_format), end_date.strftime(date_format)))
        
    return interval_list

pair_list = get_date_pairs(start_date_str,end_date_str,date_format)

def get_tick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks):
    # fixes the error where the api crashes the kernel
    date_format = '%d %b %Y'
    pair_list = get_date_pairs(start_date_str,end_date_str,date_format)
    df_list = []
    for pair in pair_list:
        print(pair[0],pair[1])
        df_list.append(get_tick_data(pair[0],pair[1]))
    df = pd.concat(df_list)
    df.to_pickle(pickle_file_name_ticks)
    return df

def get_candlestick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks,timeframe,pickle_file_name_ohlc):
    df = get_tick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks)
    ohlc_df = tick_to_ohlc(df,timeframe,pickle_file_name_ohlc)
    return df,ohlc_df

def plot_data(ohlc_df):
    # Create a candlestick chart using mplfinance
    mpf.plot(ohlc_df, type='candle', title='Candlestick Chart',tight_layout=True, ylabel='Price', figratio=(15, 10),figsize=(15,10))
    plt.show()