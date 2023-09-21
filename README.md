# bp-Trading Practicum

## Overview

This project Goal is to optimise a decision matrix to build a ladder trading strategy on EURUSD. The README will guide you through the the provided Python scripts and Jupyter notebooks.

## Prerequisites

- Python 3.x
- Jupyter Notebook

## Installation

1. **Place all .py in the Directory**: Ensure all .py files are in the same directory as your Jupyter notebooks.

2. **Install Packages**: Run the following command in your terminal to install the required packages:
keep in mind the requirements.txt will be continuously updated with every commit

    ```bash
    pip install -r requirements.txt
    ```
## FX Data Utility: Tick Data and OHLC Transformation 

`import Data_for_bp as dfb`

This module focuses on fetching tick-level foreign exchange (FX) data from Dukascopy and subsequently transforming this high-frequency data into OHLC (Open, High, Low, Close) formats. This format is widely used for traditional technical analysis and visual representation in quantitative trading.

## Data Extraction and Ladder Strategy Functions Overview

- **`get_tick_data(start_date, finish_date) -> DataFrame`**: 
  - Retrieves tick-level FX data for a specific time range.
  - Computes a mid price for the FX instrument using bid and ask prices.

- **`tick_to_ohlc(tick_df: DataFrame, timeframe: str, pickle_file_name_ohlc=None) -> DataFrame`**: 
  - Converts tick-level data to OHLC format for a designated timeframe.
  - Provides the option to save the resulting data to a pickle file.

- **`get_candlestick_data(start_date, finish_date, timeframe: str) -> Tuple[DataFrame, DataFrame]`**: 
  - A utility that fetches tick-level data and then converts it to OHLC for a specific timeframe.

- **`get_date_pairs(start_date_str, end_date_str, date_format) -> List[Tuple[str, str]]`**: 
  - Divides a broader time interval into one-month segments. This prevents excessive API calls and potential system crashes.

- **`get_tick_data_optimised(start_date_str, end_date_str, pickle_file_name_ticks=None) -> DataFrame`**: 
  - Optimized method that fetches tick-level data in monthly intervals and amalgamates them.

- **`get_candlestick_data_optimised(start_date_str, end_date_str, timeframe, pickle_file_name_ticks=None, pickle_file_name_ohlc=None) -> Tuple[DataFrame, DataFrame]`**: 
  - Optimized function to derive OHLC data from tick-level data in segments.

- **`plot_data(ohlc_df, title='candlestick chart')`**: 
  - Renders the OHLC data as a candlestick chart.

### **`year_order(start_date, end_date) -> dict`**:
  - Takes in a date range and, if across multiple years, splits into year pairs.
  - Returns a dictionary with years as keys and date ranges as values.

### **`data_gather_from_files(start_date, end_date, file_path='Data for Practicum 2') -> DataFrame`**:
  - Extracts data from various files based on the given date range.
  - Returns the data as a single DataFrame.

### **`ladderize_open(tick_data: Series, grid_size: float) -> Series`**:
  - Converts tick data into a ladderized format using the open price method.
  - Returns ladderized data as a Series.

### **`ladderize_absolute(tick_data: Series, grid_size: float) -> Series`**:
  - Converts tick data into a ladderized format using the absolute price method.
  - Returns ladderized data as a Series.

### **`plot_colored_ladder(ladderized_data: Series)`**:
  - Plots the ladderized data with color coding based on price movement.

### **`plot_ladderized(start_date, end_date, grid_size=0.0005, ladderize_function=ladderize_open)`**:
  - Loads tick data and plots it alongside the ladderized data for visualization.

### **`filter_jumps(ladderized_data: Series) -> Series`**:
  - Filters ladderized data to keep only the changes in price.
  - Returns a Series containing only the data points where there's a change.

### **`aggregate_differences(jumps: Series, lot_size=1) -> Series`**:
  - Aggregates the position for buy/sell signals based on ladder jumps.
  - Returns a Series representing the aggregated position.

### **`plot_jumps(ladderized_data: Series)`**:
  - Plots the ladderized data with jumps and the aggregated position.

### **`convert_to_grid_binomial_data(tick_data, grid_size, ladderized_function, ladder_depth=10) -> Tuple[Series, Series]`**:
  - Converts raw tick data into a binomial grid representation using a ladderized function.
  - Useful for representing price movements in a binary format for strategy development.

### **`velocity(data, grid_sizing, indicator_scale=5) -> np.array`**:
  - Computes the velocity of price movements.
  - Useful for identifying momentum in price data.

### **`acceleration(data, grid_sizing, indicator_scale=5) -> np.array`**:
  - Computes the acceleration of price movements.
  - Useful for identifying changes in momentum.

### **`apply_depth_constraint(data, depth=10) -> np.array`**:
  - Applies a depth constraint to binomial data.
  - Ensures that cumulative ladder movements do not exceed a specified depth.

### **`build_lot_sizing(lot_sizing, binomial_data, multiplier=1, indicator_data=[], min_lot_size=1000) -> np.array`**:
  - Computes the lot sizing for trading based on binomial data and optional indicator data.
  - Useful for determining position sizes based on price movement and other indicators.

### **`indicator_prep(data, grid_sizing, lookback=200, Type='v', indicator_scale=5) -> np.array`**:
  - Prepares an indicator (velocity or acceleration) based on ladderized tick data.
  - Useful for generating signals for trading decisions.

### **`plot_trades(grid_jumps, R_PNL, U_PNL, N, lookback=200)`**:
  - Plots trading data including buy/sell points, lot sizes held, and PNL.
  - Useful for visualizing trading decisions and outcomes.

### **`format_df(df) -> DataFrame`**:
  - Formats a DataFrame for display.
  - Useful for presenting trading data in a readable format.

### **`run_strategy_total_PNL(...)`**:
  - Runs a continuous trading strategy based on ladderized tick data.
  - Computes and returns PNL, position, and trade details for each time step.

### **`run_strategy_eval(...)`**:
  - Evaluates and visualizes a trading strategy based on ladderized tick data and optional indicators.
  - Computes and returns trade details, PNL, position, and other metrics for each time step.

### **`run_strategy_optimised(...)`**:
  - Optimized function to evaluate a trading strategy based on ladderized tick data and optional indicators.
  - Computes and returns key metrics like PNL, minimum unrealized PNL, maximum position, and realized PNL.

### Dependencies

- findatapy
- pandas
- datetime
- dateutil
- matplotlib
- mplfinance

**Usage**:
After cloning this repository, import the module and invoke the appropriate function to fetch tick data or OHLC data for the FX instrument and time period of your choice. This utility is specifically optimized for procuring larger datasets without overburdening the API.

## Data Analysis 

Guide to the Notebook Data_analysis 

**Usage**:
After cloning this repository, Make sure to add necessary files containing daily timeframe candlestick data or import from file titled Data for practicum 2 on box and add it to the same location.


