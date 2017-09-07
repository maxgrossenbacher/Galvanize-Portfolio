import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def pickle(filepath):
    if os.path.exists('dataframe.pkl'):
        return pd.read_pickle('dataframe.pkl')
    else:
        cols = ['SalesID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade','MachineHoursCurrentMeter',
       'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
       'fiModelSeries', 'fiModelDescriptor',
       'fiProductClassDesc', 'state', 'ProductGroupDesc',
       'Drive_System', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Differential_Type', 'Steering_Controls','Transmission', 'MachineID', 'UsageBand']
        df = pd.read_csv(filepath)
        df.drop(cols, inplace=True, axis=1)
        # keep ['ProductSize', 'ProductGroup',
        # 'Enclosure', 'Hydraulics_Flow', 'Travel_Controls',
        # 'Transmission_None or Unspecified', 'Transmission_Powershift',
        # 'Transmission_Powershuttle', 'Transmission_Standard']
        return df.sample(frac=.0005, random_state=123)

def eda():
    cols = df.columns
    df.shape # (401125, 53)
    df.info() # no null values
    df.describe() # find stats on continuous features
    df.corr()

def eda_plots(df):
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    df.boxplot('SalePrice')
    # hist(df['SalePrice'])

def clean_df(df):
    # 'MachineID'

    # 'UsageBand' map ranked values ['Low', 'High', 'Medium', nan]
    # df['UsageBand'].fillna('None or Unspecified', inplace=True)
    # ranks = {'Low':1, 'High':3, 'Medium':2, 'None or Unspecified':0}
    # df['UsageBand'] = df['UsageBand'].map(ranks)

    # 'ProductSize' ['Mini', nan, 'Large / Medium', 'Medium', 'Compact', 'Small', 'Large']
    df['ProductSize'].fillna('None or Unspecified', inplace=True)
    ranks = {'Mini':1, 'Compact':2, 'Small':3, 'Medium':4, 'Large / Medium':5, 'Large':6, 'None or Unspecified':0}
    df['ProductSize'] = df['ProductSize'].map(ranks)

    # 'ProductGroup'
    df = pd.get_dummies(df, columns=['ProductGroup'], drop_first=True)

    # 'Enclosure' ['EROPS w AC', 'OROPS', 'EROPS', nan, 'EROPS AC', 'NO ROPS', 'None or Unspecified']
    df['Enclosure'].fillna('None or Unspecified', inplace=True)
    cats = {'EROPS w AC':1, 'OROPS':1, 'EROPS':1, 'EROPS AC':1, 'NO ROPS':0, 'None or Unspecified':0}
    df['Enclosure'] = df['Enclosure'].map(cats)

    # 'Hydraulics_Flow'
    df = pd.get_dummies(df, columns=['Hydraulics_Flow'], dummy_na=True, drop_first=True)

    # 'Travel_Controls' [nan, 'None or Unspecified', 'Finger Tip', 'Differential Steer']
    df = pd.get_dummies(df, columns=['Travel_Controls'], dummy_na=True, drop_first=True)

    # 'Transmission' [nan, 'Standard', 'Powershift', 'None or Unspecified', 'Hydrostatic', 'Powershuttle']
    # df['Transmission'].fillna('None or Unspecified', inplace=True)
    # df = pd.get_dummies(df, columns=['Transmission'], drop_first=True)

    return df

def pickle_clean_df(df):
    os.system("rm %s"%'dataframe.pkl')
    df.to_pickle('dataframe.pkl')

def split_data(df):
    y = df.pop('SalePrice')
    X = df.values
    # if using statsmodels...
    # X = sm.add_constant(X)
    return train_test_split(X, y, test_size=0.25)


if __name__ == '__main__':
    df = pickle('data/Train.csv')
    # eda()
    # eda_plots(df)
    # df = clean_df(df)
    # pickle_clean_df(df)
    # X_train, X_test, y_train, y_test = split_data(df)
