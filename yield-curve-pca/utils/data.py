import sqlite3
import pandas as pd

def get_yc_data(start, end, weekly):
    df = pd.read_csv('yield_curve_history.csv')
    df = df[(df['date'] >= start) & (df['date'] <= end)]
    df = df.dropna(axis = 1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Sample for weekly
    if weekly:
        df = df.resample('W-FRI').apply(lambda x: x.loc[x.index.max()] 
                if not x.empty else None)  # Get last available day in that week
    
    # Rename columns
    def tenor_to_years(tenor):    
        if tenor.endswith('m'):
            return int(tenor[:-1]) / 12.0
        elif tenor.endswith('y'):
            return int(tenor[:-1]) * 1.0
        else:
            raise ValueError(f"Unrecognized tenor format: {tenor}")
    df.columns = [tenor_to_years(c) for c in df.columns]
    return df