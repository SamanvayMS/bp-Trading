__author__ = "samanvayms"

"""
base functions for Data Pulling, Preparation, Visualisation , EDA , Strategy Development and Evaluation
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

def get_tick_data(start_date,finish_date,currency_pair='EURUSD'):
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
                                tickers=[currency_pair],
                                vendor_tickers=[currency_pair])
    df = market.fetch_market(md_request)
    df['mid'] = (df[currency_pair+'.ask'] + df[currency_pair+'.bid']) / 2.0
    
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
    ohlc_df = tick_df['mid'].resample(timeframe).ohlc()
    
    # Drop rows where all values are NaN (which may happen in less active trading periods)
    ohlc_df.dropna(how='all', inplace=True)
    
    # write to pickle file
    if ohlc_df is not None and pickle_file_name_ohlc is not None:
        ohlc_df.to_pickle(pickle_file_name_ohlc)
    return ohlc_df

def get_candlestick_data(start_date,finish_date,timeframe: str,currency_pair='EURUSD'):
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
    tick_data = get_tick_data(start_date,finish_date,currency_pair)
    ohlc_df = tick_to_ohlc(tick_data, timeframe)
    return ohlc_df,tick_data

def get_date_pairs(start_date_str,end_date_str,date_format='%d %b %Y',interval = 1):
    # Parse the start and end dates
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)

    # Initialize an empty list to hold the interval pairs
    interval_list = []

    # Increment by one month until we reach or surpass the end date
    current_date = start_date
    next_date = current_date + relativedelta(months=interval)
    while next_date <= end_date:
        interval_list.append((current_date.strftime(date_format), next_date.strftime(date_format)))
        current_date = next_date
        next_date += relativedelta(months=interval)

    # Add the remaining interval if there are extra days left
    if current_date != end_date:
        interval_list.append((current_date.strftime(date_format), end_date.strftime(date_format)))
        
    return interval_list

def get_tick_data_optimised(start_date_str,end_date_str,pickle_file_name_ticks=None,currency_pair='EURUSD'):
    # fixes the error where the api crashes the kernel
    date_format = '%d %b %Y'
    pair_list = get_date_pairs(start_date_str,end_date_str,date_format)
    df_list = []
    for pair in pair_list:
        print(pair[0],pair[1])
        df_list.append(get_tick_data(pair[0],pair[1],currency_pair))
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
def data_gather_from_files(start_date,end_date,file_path='Data for Practicum 2',currency_pair='EURUSD'):
    year_dict = year_order(start_date,end_date)
    full_df = pd.DataFrame()
    for year in year_dict.keys():
        year_df = pd.read_pickle(f'{file_path}/{currency_pair}/ticks_' + str(year) + '.pkl')
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
            
def plot_ladderized(start_date, end_date, grid_size=0.0005, ladderize_function=ladderize_open,currency_pair='EURUSD'):
    # Load the tick data
    tick_data = get_tick_data(start_date,end_date,currency_pair)['mid']

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

# Convert to Binomial data

def convert_to_grid_binomial_data(tick_data,grid_size,ladderized_function):
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
    return jumps,binomial_data

def loop_to_scale_lots(binomial_data, positions, lot_sizing, multiplier, indicator_data):
    """
    Helper function to scale lot sizes based on market direction and binomial steps.

    Parameters:
    - binomial_data (array): Binomial representation of market moves (1 for upward, -1 for downward).
    - positions (array): The current state of positions.
    - lot_sizing (float): Initial lot size.
    - multiplier (float): Factor to increase the lot size when market direction is opposite to binomial step.
    - indicator_data (array): Data indicating market direction.

    Returns:
    - array: Updated lot sizes based on binomial and market direction.
    """
    for i in range(1, len(binomial_data)):
        if indicator_data[i] * binomial_data[i] < 0:  # Opposite direction
            positions[i] = positions[i-1] * multiplier
        else:
            positions[i] = lot_sizing
    return positions

def build_lot_sizing(lot_sizing, binomial_data, multiplier=1, indicator_data=[], just_direction=True, min_lot_size=10000):
    """
    Compute the lot sizing for trading based on binomial data and an optional market indicator.

    Parameters:
    - lot_sizing (float): Base lot size.
    - binomial_data (array): Binomial representation of tick data (1 for upward, -1 for downward).
    - multiplier (float, optional): Scaling factor for lot size. Default is 1.
    - indicator_data (array, optional): Market direction data for lot scaling. Default is an empty list.
    - just_direction (bool, optional): If True, only considers the market direction for scaling. Default is True.
    - min_lot_size (int, optional): Minimum permissible lot size. Default is 10000.

    Returns:
    - array: Scaled lot sizes for each time step.
    """
    T = len(binomial_data)
    if len(indicator_data) == 0:
        return np.ones(T)*lot_sizing
    if just_direction: # Only directional data is present
        positions = np.zeros(T,dtype=np.int64)
        binomial_data = np.array(binomial_data,dtype=np.int64)
        positions = loop_to_scale_lots(binomial_data,positions,lot_sizing,multiplier,indicator_data) # refer to function for details on how it works
        return positions
    else: # other indicators are present
        scaling_criteria =  binomial_data * indicator_data  # +ve if both are same sign hence signifies movement in same direction , we do not scale in that case
        #positions = np.where(scaling_criteria > 0,1/(1+abs(scaling_criteria)),1+abs(scaling_criteria))*lot_sizing # +ve if both are same sign hence signifies movement in same direction, we scale in by inverse
        positions = np.where(scaling_criteria > 0,1,1+abs(scaling_criteria))*lot_sizing
        positions[np.isnan(positions)] = 0
        return np.round(positions/min_lot_size,0)*min_lot_size


# ****************************************************************************************************************

# Indicator Preparation Functions

import numpy as np

def velocity(data, grid_sizing, indicator_scale=5):
    """
    Compute the velocity of a time series.
    
    Parameters:
    - data (Series): The input time series data.
    - grid_sizing (float): Grid size to normalize the difference.
    - indicator_scale (float, optional): A scale factor. Default is 5.
    
    Returns:
    - Series: Scaled velocity of the time series.
    """
    return np.round(indicator_scale * data.diff() / data / grid_sizing, 1)

def acceleration(data, grid_sizing, indicator_scale=5):
    """
    Compute the acceleration of a time series.
    
    Parameters:
    - data (Series): The input time series data.
    - grid_sizing (float): Grid size to normalize the difference.
    - indicator_scale (float, optional): A scale factor. Default is 5.
    
    Returns:
    - Series: Scaled acceleration of the time series.
    """
    return np.round(indicator_scale * data.diff().diff() / data / grid_sizing, 1)

def direction(data):
    """
    Compute the direction of a time series.
    
    Parameters:
    - data (Series): The input time series data.
    
    Returns:
    - Series: Direction of the time series. Values are -1, 0, or 1.
    """
    return np.sign(data.diff())

def indicator_prep(data, grid_sizing, lookback=200, Type='d', indicator_scale=5):
    """
    Prepare the indicator for a time series based on type ('a' for acceleration, 'v' for velocity, or 'd' for direction).
    
    Parameters:
    - data (Series): The input time series data.
    - grid_sizing (float): Grid size to normalize the difference.
    - lookback (int, optional): Number of periods for the exponential moving average. Default is 200.
    - Type (str, optional): Type of the indicator ('a', 'v', or 'd'). Default is 'd'.
    - indicator_scale (float, optional): A scale factor. Default is 5.
    
    Returns:
    - Series: The prepared indicator.
    """
    # Exponential moving average calculation
    ema_data = data.ewm(span=lookback).mean()
    
    if Type == 'a':
        data = acceleration(ema_data, grid_sizing, indicator_scale).shift(1)
    elif Type == 'v':
        data = velocity(ema_data, grid_sizing, indicator_scale).shift(1) 
    else:
        data = direction(ema_data).shift(1)
    
    # Fill NaN values with 0
    data[np.isnan(data)] = 0
    return data


# Plot the results
def plot_indicator_graph(jumps,grid_size,lookback=10):
    jump_ema = jumps.ewm(span=lookback).mean()
    fig,axs = plt.subplots(4,1,figsize=(15,15))
    axs[0].plot(jumps.values, label='jumps', linestyle='--')
    axs[0].plot(jump_ema.values, label='ema_jumps', linestyle='-')
    #axs[0].plot(ladderized_data, label='Ladderized Data',alpha=0.5)
    axs[0].legend()
    axs[1].plot(direction(jump_ema).values, label='ema_direction')
    axs[1].legend()
    axs[2].plot(velocity(jump_ema,grid_size).values, label='ema_velocity')
    axs[2].legend()
    axs[3].plot(acceleration(jump_ema,grid_size).values, label='ema_acceleration')
    axs[3].legend()
    plt.show()
# ****************************************************************************************************************
# ****************************************************************************************************************

# strategy evaluation functions

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

def run_strategy_eval(tick_data, grid_sizing, lot_sizing,
                      ladder_function=ladderize_absolute_optimised, multiplier=1,
                      indicator_type='d', indicator_scale=5, lookback=200,
                      print_trade_book=False, trade_plot=False):
    """
    Evaluate a trading strategy based on ladderized tick data and optional indicators.
    
    Parameters:
    ----------
    tick_data : pd.Series or np.array
        Raw tick data to be processed.
    
    grid_sizing : float
        Size of the grid to map the tick data.
        
    lot_sizing : float
        Base lot size for trading.
    
    ladder_function : callable, optional
        Function to ladderize the tick data based on the grid size.
        Default is ladderize_absolute_optimised.
        
    multiplier : float, optional
        Multiplier to scale the lot size. Default is 1.
        
    indicator_type : str, optional
        Type of the indicator to use. Default is 'd'.
        
    indicator_scale : int, optional
        Scale for the indicator. Default is 5.
    
    lookback : int, optional
        Lookback period for the indicator. Default is 200.
        
    print_trade_book : bool, optional
        Flag to print trade details for each time step. Default is False.
        
    trade_plot : bool, optional
        Flag to plot the trading data. Default is False.
    
    Returns:
    -------
    tuple
        Returns tuple containing realized and unrealized PNL, position, 
        and average price for each time step.
    
    Notes:
    -----
    The strategy computes PNL, position, and average price based on ladderized tick data and 
    optional indicators. Results can be visualized via trade book prints and plots.
    """
    
    # Initialize time benchmarks for performance evaluation
    time1 = time.time()
    grid_jumps,binomial_data = convert_to_grid_binomial_data(tick_data,grid_sizing,ladder_function)
    time2 = time.time()
    indicator_data = indicator_prep(grid_jumps,grid_sizing,lookback=lookback,Type=indicator_type,indicator_scale=indicator_scale)
    time3 = time.time()

    T = len(binomial_data)
    PNL = np.zeros(T)
    R_PNL = np.zeros(T)
    U_PNL = np.zeros(T)
    P = np.zeros(T)
    N = np.zeros(T)
    trades = pd.DataFrame({})
    
    direction = False
    if indicator_type == 'd':
        direction = True
        
    position_sizing = build_lot_sizing(lot_sizing,binomial_data,multiplier=multiplier,indicator_data=indicator_data,just_direction=direction)
    time4=time.time()
    avg_price = 0
    for t in np.arange(0,T):
        lots_in_order = - position_sizing[t] * binomial_data[t]
        N[t] = N[t-1] + lots_in_order
        P[t] = P[t-1] + lots_in_order * grid_jumps[t] # cash spent
        
        # check if max_position will get breached with the trade- calculated well before the trade is executed
        if (np.abs(P[t]) >= 10e6):
            print('position limit breached at {},'.format(t))
            print('position rest to P[t-1] = {}'.format(P[t-1]))
            lots_in_order = 0
            N[t] = N[t-1]
            P[t] = P[t-1]
        
        # we have 2 different situation to consider:
        # we close existing positions
        if ((N[t-1]<0) and (N[t]>N[t-1])) or ((N[t-1]>0) and (N[t]<N[t-1])): # if we are closing a position in short and long direction
            if N[t-1]*N[t] < 0: # if we are looking at the order being partially filled
                R_PNL[t] = R_PNL[t-1] + (grid_jumps[t] - avg_price) * (N[t-1]) # realised PNL
                # update avg price based on partially opened orders 
                avg_price = grid_jumps[t]
            else:
                R_PNL[t] = R_PNL[t-1] - (grid_jumps[t] - avg_price) * lots_in_order # realised PNL
        # else if we are only adding to our positions 
        else: # if we are opening a position or adding to an existing position
            R_PNL[t] = R_PNL[t-1] # realised PNL remains the same
            # since we are adding to an existing position we need to update the avg price
            if N[t]!=0: # to avoid divide by zero error
                avg_price = (avg_price * N[t-1] + grid_jumps[t]*lots_in_order)/N[t] # update avg price
            # if N[t] == 0 we are closing a position and hence avg price is not updated

        U_PNL[t] = (grid_jumps[t] - avg_price) * N[t] # unrealised PNL
        PNL[t] = PNL[t-1] + N[t-1] * (grid_jumps[t] - grid_jumps[t-1])
        
        # if unrealised PNL is less than -150k we close all positions and reset
        if U_PNL[t] < -150e3:
            print('closing all positions at t = {}'.format(t))
            R_PNL[t] = R_PNL[t] - 150e3
            U_PNL[t] = 0
            PNL[t] = R_PNL[t]
            avg_price = 0
            N[t] = 0
            P[t] = 0

        PNL[t]=np.round(PNL[t],4)
        U_PNL[t]=np.round(U_PNL[t],4)
        R_PNL[t]=np.round(R_PNL[t],4)
        
        
        # checks if profit calculations are right
        error = np.round(PNL[t] - R_PNL[t] - U_PNL[t],2) # total profit must be equal to cumulative sum of realised + unrealised profit
        if (error!=0):
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
        trade = pd.Series({'t':t,'price':grid_jumps[t],'lots_in_order':lots_in_order,'Previous_lots':N[t-1],'current_lots':N[t],'position':P[t],'R_PNL':R_PNL[t],'U_PNL':U_PNL[t],'PNL':PNL[t],'average_price':avg_price})
        trades = pd.concat([trades, trade.to_frame().T],ignore_index=True)
    time5 = time.time()
    
    # Print benchmarking results
    print('Ladderization time:', time2 - time1)
    print('Indicator preparation time:', time3 - time2)
    print('Position sizing time:', time4 - time3)
    print('PNL calculation time:', time5 - time4)
    
    # Plot trade details if required
    if trade_plot:
        plot_trades(grid_jumps, R_PNL, U_PNL, N, lookback=lookback)
    
    return PNL, R_PNL, U_PNL, N, P, trades

def run_strategy_optimised(tick_data, grid_sizing, lot_sizing,
                           ladder_function=ladderize_absolute_optimised, multiplier=1,
                           indicator_type='d', indicator_scale=5, lookback=200):
    """
    Execute a quant trading strategy using provided tick data and parameters.

    Parameters:
    - tick_data : DataFrame
        The historical price data on which the strategy will be executed.
    - grid_sizing : int
        Size of the grid to use in strategy.
    - lot_sizing : float
        Number of lots to trade in strategy.
    - ladder_function : function, optional
        Function to ladderize data, default is ladderize_absolute_optimised.
    - multiplier : int, optional
        Multiplier to adjust lot sizing, default is 1.
    - indicator_type : str, optional
        Type of indicator to use ('d' or other), default is 'd'.
    - indicator_scale : int, optional
        Scale of the indicator, default is 5.
    - lookback : int, optional
        Lookback period for the strategy, default is 200.

    Returns:
    - max_loss : float
        Maximum loss incurred during strategy execution.
    - min_U_PNL : float
        Minimum unrealised PNL during strategy execution.
    - max_position : int
        Maximum position size during strategy execution.
    - R_PNL : float
        Realised PNL at the end of the strategy.
    - PNL : float
        Total PNL at the end of the strategy.
    """
    
    grid_jumps,binomial_data = convert_to_grid_binomial_data(tick_data,grid_sizing,ladder_function)
    indicator_data = indicator_prep(grid_jumps,grid_sizing,lookback=lookback,Type=indicator_type,indicator_scale=indicator_scale)

    T = len(binomial_data)
    
    current_lots = 0
    previous_lots = 0
    R_PNL = 0
    U_PNL = 0
    PNL = 0
    current_position = 0
    previous_position = 0
    
    direction = False
    if indicator_type == 'd':
        direction = True

    position_sizing = build_lot_sizing(lot_sizing,binomial_data,multiplier=multiplier,indicator_data=indicator_data,just_direction=direction)
    
    avg_price = 0
    max_position = 0
    min_U_PNL = 0
    max_loss = 0
    count = 0
    std = 0
    
    for t in np.arange(0,T):
        lots_in_order = - position_sizing[t] * binomial_data[t]
        previous_lots = current_lots
        previous_position = current_position
        current_lots = previous_lots + lots_in_order
        current_position = previous_position + lots_in_order * grid_jumps[t] # cash spent
        
        # check if max_position will get breached with the trade- calculated well before the trade is executed
        if (np.abs(current_position)) >= 10e6:
            lots_in_order = 0
            current_lots = previous_lots
            current_position = previous_position

        # we have 2 different situation to consider:
        # we close existing positions
        if ((previous_lots<0) and (current_lots>previous_lots)) or ((previous_lots>0) and (current_lots<previous_lots)): # if we are closing a position in short and long direction
            if previous_lots*current_lots < 0: # if we are looking at the order being partially filled
                R_PNL += (grid_jumps[t] - avg_price) * (previous_lots) # realised PNL
                # update avg price based on partially opened orders 
                avg_price = grid_jumps[t]
            else:
                R_PNL -= (grid_jumps[t] - avg_price) * lots_in_order # realised PNL
        # else if we are only adding to our positions 
        else: # if we are opening a position or adding to an existing position
            if current_lots!=0: # to avoid divide by zero error
                avg_price = (avg_price * previous_lots + grid_jumps[t]*lots_in_order)/current_lots # update avg price
            # if N[t] == 0 we are closing a position and hence avg price is not updated

        PNL = PNL + previous_lots * (grid_jumps[t] - grid_jumps[t-1])
        
        U_PNL = (grid_jumps[t] - avg_price) * current_lots # unrealised PNL

        # if unrealised PNL is less than -150k we close all positions and reset
        if U_PNL < -150e3:
            R_PNL = R_PNL - 150e3
            U_PNL = 0
            PNL = R_PNL
            avg_price = 0
            current_lots = 0
            current_position = 0

        PNL=np.round(PNL,4)
        U_PNL=np.round(U_PNL,4)
        R_PNL=np.round(R_PNL,4)

        if PNL < max_loss:
            max_loss = PNL

        std = np.sqrt((count*(std**2) + (U_PNL**2))/(count+1))
        count = count + 1
    return max_loss, R_PNL , PNL , std


# ****************************************************************************************************************
# ****************************************************************************************************************

# strategy optimization functions
