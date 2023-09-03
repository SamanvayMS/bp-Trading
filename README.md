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

## Data Gathering

### Overview

To collect the necessary candlestick data, utilize the `get_candlestick_data` function from `Data_for_bp.py`.
    ```bash
    from Data_for_bp import get_candlestick_data
    ```

### Parameters

- **`start_date`**: Start date for data collection in 'dd-mmm-yyyy' format.(like '14 Jun 2016')
- **`end_date`**: End date for data collection in 'dd-mmm-yyyy' format.(like '17 Jun 2016')

- **`granularity`**: Time intervals for data collection. Options are:

    - 1 minute: `1T`
    - 1 hour: `1H`
    - 1 day: `1D`

### Instructions

1. Open the Jupyter notebook `Data_gathering.ipynb`.
2. Follow the notebook to build and utilize `Data_for_bp.py`.
