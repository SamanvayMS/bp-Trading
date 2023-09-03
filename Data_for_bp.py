__author__ = "samanvayms"

# modified from https://github.com/cuemacro/findatapy/blob/master/findatapy_examples/dukascopy_example.py 
# Author for original - saeedamen

from findatapy.market import Market, MarketDataRequest, MarketDataGenerator
import pandas as pd

def get_tick_data(start_date,finish_date):
    market = Market(market_data_generator=MarketDataGenerator())

    # first we can do it by defining all the vendor fields, tickers etc. so we bypass the configuration file
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

def tick_to_ohlc(tick_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    '''
    Convert tick data to OHLC data.
    
    timeframe format:
    1 min = 1T
    1 Hr = 1H
    1 Day = 1D
    
    '''
    # Assuming the DataFrame is indexed by timestamp and has a 'Mid' column
    # for the mid prices. Adapt as necessary.

    # Resample the tick data to OHLC data using the specified timeframe
    ohlc_df = tick_df['EURUSD.mid'].resample(granularity).ohlc()
    
    # Drop rows where all values are NaN (which may happen in less active trading periods)
    ohlc_df.dropna(how='all', inplace=True)
    
    return ohlc_df

def get_candlestick_data(start_date,finish_date,granularity: str):
    '''get candlestick data from start_date to finish_date with timeframe
    
    date format: 'dd mmm yyyy' like '14 Jun 2016'
    timeframe format:
    1 min = 1T
    1 Hr = 1H
    1 Day = 1D
    '''
    tick_data = get_tick_data(start_date,finish_date)
    ohlc_df = tick_to_ohlc(tick_data, granularity)
    return ohlc_df