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

### Functions Overview

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


