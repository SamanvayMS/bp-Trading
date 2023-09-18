__author__ = "samanvayms"

# modified from https://github.com/cuemacro/findatapy/blob/master/findatapy_examples/dukascopy_example.py 
# Author for original - saeedamen
"""
This module provides functionalities for fetching, processing, and visualizing tick data for trading purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import mplfinance as mpf 
from findatapy.market import Market, MarketDataRequest, MarketDataGenerator


def get_tick_data(start_date,finish_date):
    """
    Fetch tick data for a given date range.

    Parameters:
    - start_date: str, start date in the format 'dd mmm yyyy'
    - finish_date: str, end date in the format 'dd mmm yyyy'

    Returns:
    - df: DataFrame, tick data with bid, ask, and mid columns
    """
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

def tick_to_ohlc(tick_df: pd.DataFrame, timeframe: str,pickle_file_name_ohlc=None) -> pd.DataFrame:
    """
    Convert tick data to OHLC (Open-High-Low-Close) format.

    Parameters:
    - tick_df: DataFrame, tick data with a mid column
    - timeframe: str, resampling frequency ('1T' for 1 minute, '1H' for 1 hour, etc.)
    - pickle_file_name_ohlc: str, optional, filename to save the OHLC data as a pickle file

    Returns:
    - ohlc_df: DataFrame, OHLC data
    """
    
    # Assuming the DataFrame is indexed by timestamp and has a 'Mid' column
    # for the mid prices. Adapt as necessary.

    # Resample the tick data to OHLC data using the specified timeframe
    ohlc_df = tick_df['EURUSD.mid'].resample(timeframe).ohlc()
    
    # Drop rows where all values are NaN (which may happen in less active trading periods)
    ohlc_df.dropna(how='all', inplace=True)
    
    # write to pickle file
    if ohlc_df is not None and pickle_file_name_ohlc is not None:
        ohlc_df.to_pickle(pickle_file_name_ohlc)
    return ohlc_df

def get_candlestick_data(start_date,finish_date,timeframe: str):
    """
    Fetch tick data and convert it to candlestick (OHLC) format.

    Parameters:
    - start_date: str, start date in the format 'dd mmm yyyy'
    - finish_date: str, end date in the format 'dd mmm yyyy'
    - timeframe: str, resampling frequency ('1T' for 1 minute, '1H' for 1 hour, etc.)

    Returns:
    - ohlc_df: DataFrame, OHLC data
    - tick_data: DataFrame, raw tick data
    """
    tick_data = get_tick_data(start_date,finish_date)
    ohlc_df = tick_to_ohlc(tick_data, timeframe)
    return ohlc_df,tick_data

def get_date_pairs(start_date_str,end_date_str,date_format):
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

def get_tick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks=None):
    # fixes the error where the api crashes the kernel
    date_format = '%d %b %Y'
    pair_list = get_date_pairs(start_date_str,end_date_str,date_format)
    df_list = []
    for pair in pair_list:
        print(pair[0],pair[1])
        df_list.append(get_tick_data(pair[0],pair[1]))
    df = pd.concat(df_list)
    if pickle_file_name_ticks is not None:
        df.to_pickle(pickle_file_name_ticks)
    return df

def get_candlestick_data_optimised(start_date_str,end_date_str,timeframe,pickle_file_name_ticks=None,pickle_file_name_ohlc=None):
    df = get_tick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks)
    ohlc_df = tick_to_ohlc(df,timeframe,pickle_file_name_ohlc)
    return df,ohlc_df

def plot_data(ohlc_df,title='canlestick chart'):
    # Create a candlestick chart using mplfinance
    mpf.plot(ohlc_df, type='candle', title=title,tight_layout=True, ylabel='Price', figratio=(15, 10),figsize=(15,7))
    plt.show()

# function that takes in a adate range and if across multiple years splits into year pairs.
def year_order(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date.year == end_date.year:
        return {start_date.year: [start_date, end_date]}
    else:
        year_dict = {}
        for year in range(start_date.year, end_date.year + 1):
            if year == start_date.year:
                year_dict[year] = [start_date, pd.to_datetime(str(year+1) + '-12-31')]
            elif year == end_date.year:
                year_dict[year] = [pd.to_datetime(str(year) + '-01-01'), end_date]
            else:
                year_dict[year] = [pd.to_datetime(str(year) + '-01-01'), pd.to_datetime(str(year) + '-12-31')]
        return year_dict
    
# extracts data from various files based on date range given and then returns it as a single dataframe
def data_gather_from_files(start_date,end_date,file_path='Data for Practicum 2'):
    year_dict = year_order(start_date,end_date)
    full_df = pd.DataFrame()
    for year in year_dict.keys():
        year_df = pd.read_pickle(file_path + '/ticks_' + str(year) + '.pkl')
        year_dict[year][0] = year_dict[year][0].tz_localize('UTC')
        year_dict[year][1] = year_dict[year][1].tz_localize('UTC')
        year_df = year_df.loc[year_dict[year][0]:year_dict[year][1]]
        full_df = pd.concat([full_df,year_df])
    return full_df

def ladderize_open(tick_data, grid_size):
    """
    Convert tick data into a ladderized format using a specified grid size. 
    This function uses the open price method.

    Parameters:
    - tick_data: Series, tick data
    - grid_size: float, size of the grid to discretize the tick data

    Returns:
    - Series, ladderized data
    """
    ladderized_data = [tick_data.iloc[0]]
    for i in range(1, len(tick_data)):
        if tick_data.iloc[i] > ladderized_data[-1] + grid_size:
            ladderized_data.append(ladderized_data[-1] + grid_size)
        elif tick_data.iloc[i] < ladderized_data[-1] - grid_size:
            ladderized_data.append(ladderized_data[-1] - grid_size)
        else:
            ladderized_data.append(ladderized_data[-1])
    # Adding the final close price
    ladderized_data[-1]=tick_data.iloc[-1]
    return pd.Series(np.round(ladderized_data,4), index=tick_data.index)

def ladderize_absolute(tick_data, grid_size):
    """
    Convert tick data into a ladderized format using a specified grid size. 
    This function uses the absolute price method.

    Parameters:
    - tick_data: Series, tick data
    - grid_size: float, size of the grid to discretize the tick data

    Returns:
    - Series, ladderized data
    """
    # Initialize ladder at the nearest rounded price level based on grid size
    ladderized_data = [(tick_data.iloc[0] / grid_size).round() * grid_size]
    for i in range(1, len(tick_data)):
        if tick_data.iloc[i] > ladderized_data[-1] + grid_size:
            ladderized_data.append(ladderized_data[-1] + grid_size)
        elif tick_data.iloc[i] < ladderized_data[-1] - grid_size:
            ladderized_data.append(ladderized_data[-1] - grid_size)
        else:
            ladderized_data.append(ladderized_data[-1])
    # Adding the final close price and changing th open price
    ladderized_data[-1]=tick_data.iloc[-1]
    ladderized_data[0]=tick_data.iloc[0]
    return pd.Series(np.round(ladderized_data,4), index=tick_data.index)

def plot_colored_ladder(ladderized_data):
    for i in range(1, len(ladderized_data)):
        if ladderized_data[i] > ladderized_data[i-1]:
            plt.plot(ladderized_data.index[i-1:i+1], ladderized_data.iloc[i-1:i+1], color='red')
        elif ladderized_data[i] < ladderized_data[i-1]:
            plt.plot(ladderized_data.index[i-1:i+1], ladderized_data.iloc[i-1:i+1], color='green')
        else:
            plt.plot(ladderized_data.index[i-1:i+1], ladderized_data.iloc[i-1:i+1], color='blue')  # Neutral color for no change
            
def plot_ladderized(start_date, end_date, grid_size=0.0005, ladderize_function=ladderize_open):
    # Load the tick data
    tick_data = get_tick_data(start_date,end_date)['EURUSD.mid']

    ladderized_data = ladderize_function(tick_data, grid_size)

    # Plot the results
    plt.figure(figsize=(15,6))
    plt.plot(tick_data, label='Tick Data',alpha=0.5)
    plot_colored_ladder(ladderized_data)
    # plt.plot(ladderized_data, label='Ladderized Data', linestyle='--')
    plt.title('Ladder Strategy Visualization for date range: {} to {}'.format(start_date, end_date))
    plt.legend()
    plt.show()
    
def filter_jumps(ladderized_data):
    """
    Filters ladderized data to keep only the changes in price.

    :param ladderized_data: A pandas Series of ladderized data.
    :return: A pandas Series containing only the data points where there's a change.
    """
    # Calculate the difference between consecutive ladderized data points
    diff = ladderized_data.diff()

    # Filter where the difference is non-zero and include the first data point
    jumps = ladderized_data[diff != 0.0]

    return jumps

def aggregate_differences(jumps,lot_size=1):
    """
    Aggregate the position for buy/sell signals.

    :param jumps: A pandas Series of ladderized data filtered for jumps.
    :return: A pandas Series representing the aggregated position.
    """
    aggregated_position = [0]  # starting from 0
    position = 0
    previous_value = jumps.values[0]
    
    for value in jumps.values[1:]:
        if value > previous_value:
            position -= lot_size  # selling one lot
        else:
            position += lot_size  # buying one lot
        aggregated_position.append(position)
        previous_value = value
    aggregated_position[-1]=0 # closing the position
    return pd.Series(aggregated_position, index=jumps.index)

def plot_jumps(ladderized_data):
    jumps = filter_jumps(ladderized_data)
    aggregated_diff = aggregate_differences(jumps)
    fig,axs = plt.subplots(2,1,figsize=(10,10))
    # Plotting the jumps
    axs[0].plot(jumps.values, label='binomial jumps', linestyle='--', alpha=0.7)
    axs[0].set_title('ladder with jumps')
    # Adding colored points for up and down movements
    previous_value = jumps.values[0]
    for idx, value in enumerate(jumps.values[1:], 1):
        if idx == len(jumps)-1:
            continue
        elif value > previous_value:
            axs[0].plot(idx, value, 'ro')  # Red point for upward movement
        elif value < previous_value:
            axs[0].plot(idx, value, 'go')  # green point for downward movement
        else:
            axs[0].plot(idx, value, 'bo') # blue point for no change
        previous_value = value
        
        # Add vertical line to all subplots
        for ax in axs:
            ax.axvline(idx, alpha=0.5, color='gray')
            
            
    axs[0].legend()
    # Plotting the aggregated differences
    axs[1].plot(aggregated_diff.values, label='position', linestyle='-', color='purple', alpha=0.8, drawstyle='steps-post')
    axs[1].legend()
    axs[1].set_title('positions')
    plt.show()

def convert_to_grid_binomial_data(tick_data,grid_size,ladderized_function):
    ladderized_data = ladderized_function(tick_data,grid_size)
    jumps = filter_jumps(ladderized_data)
    binomial_data = jumps.diff()
    binomial_data[1:] = np.where(binomial_data[1:] > 0,1,-1)
    binomial_data[0] = 0
    return jumps,binomial_data

def velocity(data,grid_sizing):
    return np.round(5*data.diff()/(data)/grid_sizing,1)

def acceleration(data,grid_sizing):
    return np.round(5*data.diff().diff()/(data)/grid_sizing,1)

def build_lot_sizing(lot_sizing,binomial_data,multiplier=1,indicator_data=[]):
    T = len(binomial_data)
    if len(indicator_data) == 0: # No indicator data
        return [lot_sizing * (multiplier**i) for i in range(T)]
    else: # Indicator data is present
        scaling_criteria =  binomial_data * indicator_data  # +ve if both are same sign hence signifies movement in same direction , we do not scale in that case
        positions = np.where(scaling_criteria > 0,1,1+abs(scaling_criteria))*lot_sizing # +ve if both are same sign hence signifies movement in same direction , we do not scale in that case
        positions[np.isnan(positions)] = 0
        return positions
    
def indicator_prep(data,grid_sizing):
    data = velocity(data.ewm(span=10).mean(),grid_sizing).shift(1) # we shift this because our lot sizing will be decided by what the values are prior not current
    data[np.isnan(data)] = 0
    return data

def plot_trades(grid_jumps,PNL,N,lookback=10):
    fig,axs = plt.subplots(3,1,figsize=(15,15))
    axs[0].plot(grid_jumps.values, label='jumps', linestyle='--')
    axs[0].plot(grid_jumps.ewm(span=lookback).mean().values, label='indicator', linestyle='-',alpha=0.6)
    for idx, i in enumerate(grid_jumps):
        if grid_jumps[idx] > grid_jumps[idx-1]:
            axs[0].plot(idx, i, 'ro')  # Red point for upward movement
        elif grid_jumps[idx] < grid_jumps[idx-1]:
            axs[0].plot(idx, i, 'go')  # Green point for downward movement
        else:
            axs[0].plot(idx, i, 'bo')  # Blue point for no change

        # Add vertical line to all subplots
        for ax in axs:
            ax.axvline(idx, alpha=0.2, color='gray')

    axs[0].set_title('buy sell points')
    axs[1].plot(N)
    # change number of digits in y axis
    axs[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axs[1].set_title('lots held')
    axs[2].plot(PNL)
    axs[2].set_title('PNL')
    plt.show()
    
def run_strategy_continuous(tick_data,grid_sizing,lot_sizing,ladder_function=ladderize_absolute,multiplier=1,indicator = False,print_trade_book=False,trade_plot=False):
    """
    Run a continuous trading strategy based on ladderized tick data.

    Parameters:
    - tick_data: Series, raw tick data
    - grid_sizing: float, grid size for ladderization
    - lot_sizing: float, initial lot size for trading
    - ladder_function: function, ladderization function (default is ladderize_absolute)
    - multiplier: float, multiplier for position sizing (default is 1)
    - indicator: bool, whether to use an indicator for position sizing (default is False)
    - print_trade_book: bool, whether to print the trade book (default is False)
    - trade_plot: bool, whether to plot the trades (default is False)

    Returns:
    - PNL: array, profit and loss for each time step
    - N: array, number of lots held at each time step
    - P: array, position value at each time step
    - trades: DataFrame, trade book
    """
    grid_jumps,binomial_data = convert_to_grid_binomial_data(tick_data,grid_sizing,ladder_function)
    lookback = 15
    indicator_data = []
    if indicator:
        indicator_data = indicator_prep(grid_jumps,grid_sizing)
    T = len(binomial_data)
    PNL = np.zeros(T)
    P = np.zeros(T)
    N = np.zeros(T)
    trades = pd.DataFrame(columns=['t','price','Previous_lots','current_lots','position','PNL'])
    position_sizing = build_lot_sizing(lot_sizing,binomial_data,multiplier=multiplier,indicator_data=indicator_data)
    for t in np.arange(0,T):
        N[t] = N[t-1] - position_sizing[t] * binomial_data[t]
        P[t] = N[t] * grid_jumps[t]
        PNL[t] = PNL[t-1] + N[t-1] * (grid_jumps[t] - grid_jumps[t-1])
        PNL[t]=np.round(PNL[t],2)
        if print_trade_book:
            print('t = {}, price={}, Previous_lots = {},  current_lots= {}, position = {}, PNL = {}'.format(t,grid_jumps[t],N[t-1],N[t],P[t],PNL[t]))
        trade = pd.Series({'t':t,'price':grid_jumps[t],'Previous_lots':N[t-1],'current_lots':N[t],'position':P[t],'PNL':PNL[t]})
        trades = pd.concat([trades, trade.to_frame().T])
    if trade_plot:
        plot_trades(grid_jumps,PNL,N,lookback=lookback)
    return PNL,N,P,trades


