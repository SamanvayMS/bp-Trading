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
import time
from numba import jit

# ****************************************************************************************************************
# Data Gathering Functions


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

# ****************************************************************************************************************
# ****************************************************************************************************************

# Data Engineering Functions

# ****************************************************************************************************************
# Ladderise Functions
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
    
@jit(nopython=True)
def ladderize_open_loop(n, tick_data, ladderized_data, grid_size):
    ladderized_data[0] = tick_data[0]
    for i in range(1, n):
        last_ladder = ladderized_data[i - 1]
        tick = tick_data[i]
        
        if tick > last_ladder + grid_size:
            ladderized_data[i] = last_ladder + grid_size
        elif tick < last_ladder - grid_size:
            ladderized_data[i] = last_ladder - grid_size
        else:
            ladderized_data[i] = last_ladder

def ladderize_open_optimised(tick_data, grid_size):
    """
    Convert tick data into step-based data using a specified grid size.

    :param tick_data: A pandas Series of tick data.
    :param grid_size: The size of the grid to discretize the tick data.
    :return: A pandas Series of ladderized data.
    """
    n = len(tick_data)
    ladderized_data = np.empty(n, dtype=np.float64)
    tick_data_np = tick_data.values.astype(np.float64)

    ladderize_open_loop(n, tick_data_np, ladderized_data, grid_size)

    # Overwrite the last tick to the exact closing price
    ladderized_data[-1] = tick_data.iloc[-1]
    
    return pd.Series(np.round(ladderized_data, 4), index=tick_data.index)

@jit(nopython=True)
def ladderize_absolute_loop(n, tick_data, ladderized_data, grid_size, rounded_open):
    last_ladder = rounded_open
    for i in range(1,n):
        tick = tick_data[i]
        if tick > last_ladder + grid_size:
            last_ladder += grid_size
        elif tick < last_ladder - grid_size:
            last_ladder -= grid_size
        ladderized_data[i] = last_ladder

def ladderize_absolute_optimised(tick_data, grid_size):
    """
    Convert tick data into step-based data using a specified grid size.

    :param tick_data: A pandas Series of tick data.
    :param grid_size: The size of the grid to discretize the tick data.
    :return: A pandas Series of ladderized data.
    """
    n = len(tick_data)
    ladderized_data = np.empty(n, dtype=np.float64)
    
    # Initialize the first point to the actual opening price
    ladderized_data[0] = tick_data.iloc[0]
    
    # Round the opening price for grid calculations
    rounded_open = (tick_data.iloc[0] / grid_size).round() * grid_size
    
    tick_data_np = tick_data.values.astype(np.float64)
    
    ladderize_absolute_loop(n, tick_data_np, ladderized_data, grid_size, rounded_open)

    # Overwrite the last tick to the exact closing price
    ladderized_data[-1] = tick_data.iloc[-1]
    
    return pd.Series(np.round(ladderized_data, 4), index=tick_data.index)

# ****************************************************************************************************************

# Filter jumps and aggregate differences

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

# ****************************************************************************************************************

# Convert to Binomial data and apply Depth constraint

def apply_depth_constraint(data, depth=10):
    sum = 0
    for i,point in enumerate(data):
        sum = sum + point
        if (sum > depth) and (point > 0):
            data[i] = 0
            sum = sum - point
        elif (sum < -depth) and (point < 0):
            data[i] = 0
            sum = sum - point
        else:
            continue
    return data

def convert_to_grid_binomial_data(tick_data,grid_size,ladderized_function,ladder_depth=10):
    """
    Convert tick data to a binomial grid representation using a ladderized function.

    Parameters:
    - tick_data (pd.Series or np.array): The raw tick data to be processed.
    - grid_size (float): The size of the grid to which the tick data will be mapped.
    - ladderized_function (callable): A function that ladderizes the tick data based on the grid size.
    - ladder_depth (int, optional): The maximum depth of the ladder. Default is 10.

    Returns:
    - tuple:
        - jumps (pd.Series or np.array): The ladderized tick data after filtering jumps.
        - binomial_data (pd.Series or np.array): The binomial representation of the tick data, where 1 indicates an upward move, -1 indicates a downward move, and 0 indicates no move beyond the ladder depth.

    Notes:
    The function first ladderizes the tick data using the provided ladderized function. It then identifies and filters out jumps in the ladderized data. The differences between consecutive ladderized data points are computed to generate the binomial representation. The binomial data is then adjusted such that any cumulative sum (or ladder aggregate) beyond the specified ladder depth is set to 0.

    Example:
    Given tick_data as [100, 101, 102, 104], grid_size as 1, and ladderized_function that rounds to the nearest integer:
    The ladderized data might be [100, 101, 102, 104]
    The jumps, after filtering, might remain the same.
    The binomial data would be [0, 1, 1, 1]
    """
    ladderized_data = ladderized_function(tick_data,grid_size)
    jumps = filter_jumps(ladderized_data)
    binomial_data = jumps.diff()
    binomial_data[1:] = np.where(binomial_data[1:] > 0,1,-1) # 1 for up and -1 for down
    binomial_data[0] = 0 # first value is always 0
    binomial_data = apply_depth_constraint(binomial_data,ladder_depth)
    return jumps,binomial_data

def build_lot_sizing(lot_sizing,binomial_data,multiplier=1,indicator_data=[],min_lot_size = 1000):
    """
    Compute the lot sizing for trading based on binomial data and optional indicator data.

    Parameters:
    - lot_sizing (float): Base lot size for trading.
    - binomial_data (pd.Series or np.array): Binomial representation of tick data, where 1 indicates an upward move, -1 indicates a downward move.
    - multiplier (float, optional): Multiplier for scaling the lot size. Default is 1.
    - indicator_data (list or np.array, optional): Additional data used for scaling the lot size based on the direction of the binomial data. Default is an empty list.

    Returns:
    - np.array: Array of lot sizes for each time step.

    Notes:
    If no indicator data is provided, the function returns a list of lot sizes scaled by the multiplier for each time step.
    If indicator data is provided, the function scales the lot size based on the alignment of the binomial data and the indicator data. Specifically, if both the binomial data and the indicator data have the same sign (indicating movement in the same direction), the lot size is inversely scaled. Otherwise, it is scaled up.

    Example:
    Given lot_sizing as 10, binomial_data as [1, -1, 1], multiplier as 2, and indicator_data as [1, -1, 1]:
    The resulting lot sizes would be [10, 20, 5] (assuming rounding for simplicity).
    """
    T = len(binomial_data)
    if len(indicator_data) == 0: # No indicator data
        return [lot_sizing * (multiplier**i) for i in range(T)]
    else: # Indicator data is present
        scaling_criteria =  binomial_data * indicator_data  # +ve if both are same sign hence signifies movement in same direction, we scale in by inverse
        positions = np.where(scaling_criteria > 0,1/(1+abs(scaling_criteria)),1+abs(scaling_criteria))*lot_sizing # +ve if both are same sign hence signifies movement in same direction , we do not scale in that case
        positions[np.isnan(positions)] = 0
        return np.round(positions/min_lot_size,0)*min_lot_size

# ****************************************************************************************************************

# Indicator Preparation Functions

def velocity(data,grid_sizing,indicator_scale=5):
    return np.round(indicator_scale*data.diff()/(data)/grid_sizing,1)

def acceleration(data,grid_sizing,indicator_scale=5):
    return np.round(indicator_scale*data.diff().diff()/(data)/grid_sizing,1)

def indicator_prep(data,grid_sizing,lookback = 200,Type = 'v',indicator_scale=5):
    if Type == 'a':
        data = acceleration(data.ewm(span=lookback).mean(),grid_sizing,indicator_scale).shift(1)
    elif Type == 'v':
        data = velocity(data.ewm(span=lookback).mean(),grid_sizing,indicator_scale).shift(1) # we shift this because our lot sizing will be decided by what the values are prior not current
    else:
        print('wrong Indicator type')
        return None
    data[np.isnan(data)] = 0
    return data

# ****************************************************************************************************************
# ****************************************************************************************************************


def plot_trades(grid_jumps, R_PNL, U_PNL, N, lookback=200):
    """
    Plot trading data including buy/sell points, lot sizes held, and PNL.

    Parameters:
    - grid_jumps (pd.Series): Ladderized tick data after filtering jumps.
    - R_PNL (pd.Series or np.array): Realized Profit and Loss (PNL) over time.
    - U_PNL (pd.Series or np.array): Unrealized Profit and Loss (PNL) over time.
    - N (pd.Series or np.array): Number of lots held over time.
    - lookback (int, optional): Lookback period for the exponential moving average (EMA) indicator. Default is 200.

    Returns:
    - None: Displays a matplotlib plot with three subplots.

    Notes:
    The function generates a plot with three subplots:
    1. Buy/sell points: This subplot displays the grid jumps with an EMA overlay. Buy points are indicated with red dots, sell points with green dots, and no change with blue dots.
    2. Lots held: This subplot displays the number of lots held over time.
    3. PNL: This subplot displays the unrealized PNL, realized PNL, and total PNL over time.

    Each buy/sell/no-change point in the first subplot corresponds to a vertical line across all subplots for visual alignment.

    Example:
    Given grid_jumps as a time series of ladderized tick data, R_PNL and U_PNL as time series of realized and unrealized PNL respectively, and N as a time series of lots held, the function will display the described plots.
    """
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
    axs[1].plot(N,drawstyle='steps-post')
    # change number of digits in y axis
    axs[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axs[1].set_title('lots held')
    axs[2].plot(U_PNL,drawstyle='steps-post',label='Unrealised PNL')
    axs[2].plot(R_PNL,drawstyle='steps-post',label='Realised PNL')
    axs[2].plot(U_PNL+R_PNL,drawstyle='steps-post',label='Total PNL')
    axs[2].legend()
    axs[2].set_title('PNL')
    plt.show()
    
def format_df(df):
    # Set the display option for floats
    pd.options.display.float_format = '{:,.2f}'.format

    # Format the DataFrame
    formatted_df = df.style.format({
        't': '{:,.0f}',  # Fixed the missing quotation mark here
        'price': '{:,.3f}',
        'Previous_lots': '{:,.0f}',
        'current_lots': '{:,.0f}',  # Fixed the missing quotation mark here
        'position': '{:,.2f}',
        'Unrealised_PNL': '{:,.2f}',
        'Realized_PNL': '{:,.2f}'
    })
    
    return formatted_df

def run_strategy_eval(tick_data,grid_sizing,lot_sizing,ladder_depth=10,ladder_function=ladderize_absolute_optimised,multiplier=1,indicator = False,lookback = 200,print_trade_book=False,trade_plot=False,print_trade_df=False):
    """
    Evaluate and visualize a trading strategy based on ladderized tick data and optional indicators.

    Parameters:
    - tick_data (pd.Series or np.array): The raw tick data to be processed.
    - grid_sizing (float): The size of the grid to which the tick data will be mapped.
    - lot_sizing (float): Base lot size for trading.
    - ladder_function (callable, optional): A function that ladderizes the tick data based on the grid size. Default is ladderize_absolute.
    - multiplier (float, optional): Multiplier for scaling the lot size. Default is 1.
    - indicator (bool, optional): Flag to determine if an indicator should be used. Default is False.
    - lookback (int, optional): Lookback period for the indicator. Default is 200.
    - print_trade_book (bool, optional): Flag to print trade details for each time step. Default is False.
    - trade_plot (bool, optional): Flag to plot the trading data. Default is False.

    Returns:
    - pd.DataFrame: A dataframe containing trade details for each time step, including price, lots, position, realized and unrealized PNL, and average price.

    Notes:
    The function evaluates a trading strategy based on ladderized tick data and optional indicators. It computes the realized and unrealized PNL, position, and average price for each time step. The function can also print trade details for each time step and plot the trading data.

    The strategy is evaluated by first converting the tick data to a binomial grid representation using the provided ladder function. If an indicator is used, it is prepared based on the grid jumps and grid sizing. The position sizing is then computed based on the binomial data, lot sizing, multiplier, and optional indicator data.

    The function then iterates over each time step, computing the number of lots in order, position, realized and unrealized PNL, and average price. The results are stored in a dataframe, which is returned.

    """
    grid_jumps,binomial_data = convert_to_grid_binomial_data(tick_data,grid_sizing,ladder_function,ladder_depth)
    indicator_data = []
    if indicator:
        indicator_data = indicator_prep(grid_jumps,grid_sizing,lookback=lookback)
        

    T = len(binomial_data)
    PNL = np.zeros(T)
    R_PNL = np.zeros(T)
    U_PNL = np.zeros(T)
    P = np.zeros(T)
    N = np.zeros(T)
    trades = pd.DataFrame(columns=['t','price','Previous_lots','current_lots','position','R_PNL','U_PNL','PNL'])
    
    position_sizing = build_lot_sizing(lot_sizing,binomial_data,multiplier=multiplier,indicator_data=indicator_data)
    avg_price = 0
    for t in np.arange(0,T):
        lots_in_order = - position_sizing[t] * binomial_data[t]
        N[t] = N[t-1] + lots_in_order
        P[t] = N[t] * grid_jumps[t] 

        if (N[t]*lots_in_order <= 0) and (lots_in_order != 0):# if we are closing a position 
        # we check if the final position and the order are in opposite directions and if the order is not 0 (i.e. we are not opening a position)
        # ----- need to improve condition when we have lot size scaling. (lets say we are - ve 10 and we buy 20 we have realised PNL only for the first 10)
            R_PNL[t] = R_PNL[t-1] - (grid_jumps[t] - avg_price) * lots_in_order # realised PNL
        else: # if we are opening a position or adding to an existing position
            R_PNL[t] = R_PNL[t-1] # realised PNL remains the same
            if N[t]!=0: # since we are adding to an existing position we need to update the avg price
                avg_price = (avg_price * N[t-1] + grid_jumps[t]*lots_in_order)/N[t] # update avg price
            # if N[t] == 0 we are closing a position and hence avg price is not updated

        PNL[t] = PNL[t-1] + N[t-1] * (grid_jumps[t] - grid_jumps[t-1])
        
        U_PNL[t] = (grid_jumps[t] - avg_price) * N[t] # unrealised PNL
        
        PNL[t]=np.round(PNL[t],4)
        U_PNL[t]=np.round(U_PNL[t],4)
        R_PNL[t]=np.round(R_PNL[t],4)
        
        
        # checks if profit calculations are right
        test = True  # set to true to test
        error = np.round(PNL[t] - R_PNL[t] - U_PNL[t],2) # total profit must be equal to cumulative sum of realised + unrealised profit
        if (error!=0) and test:
            print('error at t = {}, error = {}'.format(t,error))

        
        if print_trade_book:
            print('At t = {}'.format(t))
            if t == 0:
                print('no trades')
            elif lots_in_order > 0:
                print('buy {} lots at price {}'.format(lots_in_order,grid_jumps[t]))
            else:
                print('sell {} lots at price {}'.format(-lots_in_order,grid_jumps[t]))
            print('average price = {}'.format(avg_price))
            print('cummulative realised PNL = {}'.format(R_PNL[t]))
            print('unrealised PNL = {}'.format(U_PNL[t]))
            print('PNL = {}'.format(PNL[t]))
        trade = pd.Series({'t':t,'price':grid_jumps[t],'Previous_lots':N[t-1],'current_lots':N[t],'position':P[t],'R_PNL':R_PNL[t],'U_PNL':U_PNL[t],'PNL':PNL[t],'average_price':avg_price})
        trades = pd.concat([trades, trade.to_frame().T])
    if trade_plot:
        plot_trades(grid_jumps,R_PNL,U_PNL,N,lookback=lookback)
    if print_trade_df:
        print(format_df(trades))
    return PNL,R_PNL,U_PNL,N,P

def run_strategy_optimised(tick_data,grid_sizing,lot_sizing,ladder_depth = 10,ladder_function=ladderize_absolute_optimised,multiplier=1,indicator = False,lookback = 200):

    grid_jumps,binomial_data = convert_to_grid_binomial_data(tick_data,grid_sizing,ladder_function,ladder_depth)
    indicator_data = []
    if indicator:
        indicator_data = indicator_prep(grid_jumps,grid_sizing,lookback=lookback)

    T = len(binomial_data)
    
    current_lots = 0
    previous_lots = 0
    R_PNL = 0
    U_PNL = 0
    PNL = 0
    position = 0
    position_sizing = build_lot_sizing(lot_sizing,binomial_data,multiplier=multiplier,indicator_data=indicator_data)
    avg_price = 0
    
    max_position = 0
    min_U_PNL = 0
    for t in np.arange(0,T):
        lots_in_order = - position_sizing[t] * binomial_data[t]
        previous_lots = current_lots
        current_lots = previous_lots + lots_in_order
        position = current_lots * grid_jumps[t] 

        if (current_lots*lots_in_order <= 0) and (lots_in_order != 0):# if we are closing a position 
        # we check if the final position and the order are in opposite directions and if the order is not 0 (i.e. we are not opening a position)
        # ----- need to improve condition when we have lot size scaling. (lets say we are - ve 10 and we buy 20 we have realised PNL only for the first 10)
            R_PNL -= (grid_jumps[t] - avg_price) * lots_in_order # realised PNL
        else: # if we are opening a position or adding to an existing position
            if current_lots!=0: # since we are adding to an existing position we need to update the avg price
                avg_price = (avg_price * previous_lots + grid_jumps[t]*lots_in_order)/current_lots # update avg price
            # if N[t] == 0 we are closing a position and hence avg price is not updated

        PNL = PNL + previous_lots * (grid_jumps[t] - grid_jumps[t-1])
        
        U_PNL = (grid_jumps[t] - avg_price) * current_lots # unrealised PNL

        PNL=np.round(PNL,4)
        U_PNL=np.round(U_PNL,4)
        R_PNL=np.round(R_PNL,4)
        if U_PNL < min_U_PNL:
            min_U_PNL = U_PNL
        if abs(position) > max_position:
            max_position = abs(position)
        if PNL < max_loss:
            max_loss = PNL
    return max_loss, min_U_PNL, max_position, R_PNL