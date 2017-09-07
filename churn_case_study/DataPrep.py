import pandas as pd
import numpy as np
import os, pickle
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

class DataPrepMgr(object):
    '''
    Manage the loading and cleaning of the churn data.
    '''
    version = '0.5.3'
    # 0.5.3 Fixed rated_driver and rated_rider
    # 0.5.2 Added rated_driver, rated_rider
    # 0.5.1 Drop dates
    # 0.5 Standardize the avg_dist column
    # 0.4.2 Removed active_days - leakage problem
    # 0.4.1 Cleaned up pop of y from changing self.df.
    pkl_ext = '.pkl'
    csv_ext = '.csv'

    def __init__(self, filename, split=False, standardize=False, save=True):
        '''
        Input:
            in_csv_filename: Base name of the CSV and the Pickle file holding the data.

        Output:
        '''
        print('DataPrepMgr: ', DataPrepMgr.version)
        self.save = save
        self.in_csv_filename = filename + DataPrepMgr.csv_ext
        self.out_csv_filename = filename + '_prepped_' + DataPrepMgr.csv_ext
        self.pkl_filename = filename + DataPrepMgr.pkl_ext
        self._load()

        dfx = self.df.copy(deep=True)
        self.y = dfx.pop('active')
        self.X = dfx

        if split:
            print('Splitting')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
                (self.X, self.y, stratify=self.y)

        if standardize:
            if split:
                print('Standardizing split')
                self.avg_dist_mean = self.X_train.avg_dist.mean()
                self.avg_dist_std = self.X_train.avg_dist.std()
                ad = self.X_train.pop('avg_dist')
                ad_stand = self._standardize(ad)
                self.X_train = pd.concat([self.X_train, ad_stand], axis=1)

                ad = self.X_test.pop('avg_dist')
                ad_stand = self._standardize(ad)
                self.X_test = pd.concat([self.X_test, ad_stand], axis=1)

                # print('Train: ', self.X_train.avg_dist.mean(), self.X_train.avg_dist.std())
                # print('Test: ', self.X_test.avg_dist.mean(), self.X_test.avg_dist.std())

            else:
                print('Standardizing non-split')
                self.avg_dist_mean = self.X.avg_dist.mean()
                self.avg_dist_std = self.X.avg_dist.std()

                ad = self.X.pop('avg_dist')
                ad_stand = self._standardize(ad)
                self.X = pd.concat([self.X, ad_stand], axis=1)

                # print('X: ', self.X.avg_dist.mean(), self.X.avg_dist.std())

    def _standardize(self, X):
        return (X-self.avg_dist_mean)/self.avg_dist_std

    def _load(self):
        '''
        Private function to handle the loading of the date from CSV or Pickle.
        '''
        if os.path.exists(self.pkl_filename):
            print('Loading {} pickle'.format(self.pkl_filename))
            f = open(self.pkl_filename, 'rb')
            self.df = pickle.load(f)
            f.close()
        else:
            print('Loading {} csv'.format(self.in_csv_filename))
            self.df = pd.read_csv(self.in_csv_filename)
            self._clean()
            if self.save:
                self._dump()

    def _clean(self):
        '''
        Clean up the datframe.  Transforms include:
        1) Convert last_trip_date to datetime
        2) Convert signup_date to datetime
        3) Create dummies for city
        4) Create dummies for phone
        5) Convert luxury_car_user to 1, 0
        6) Create new column for days between signup_date and last_trip_date
        7) Create output label for active.
        '''
        print('Cleaning data')
        # Convert last_trip_date and signump_date to datetime
        self.df['last_trip_date'] = pd.to_datetime(self.df['last_trip_date'])
        self.df['signup_date'] = pd.to_datetime(self.df['signup_date'])

        # Explode city
        self.df = pd.get_dummies(self.df, prefix='city', prefix_sep='_', columns=['city'], drop_first=True)

        # Explode phone
        self.df = pd.get_dummies(self.df, columns=['phone'], drop_first=True)

        # Convert luxury car to numeric
        self.df['luxury_car_user'] = self.df.apply \
                (lambda r: int(r.luxury_car_user), axis=1)

        # Calculate active_days = last_trip_date - signup_date
        # Remove this as it is data leakage problem.  This is tied in
        # with how we calculate active as all the drivers joined in january.
        # if we us this we are only dealing with when the driver joined
        # in january as a difference between active.
        self.df['active_days'] = self.df.apply \
                (lambda r: int((r.last_trip_date - r.signup_date) / np.timedelta64(1,'D')), axis=1)

        # Calculate if they are active.  last_trip_date - 2014-05-01 <= 30
        active_cut_date = pd.to_datetime('2014-05-01')
        self.df['active'] = self.df.apply \
                (lambda r: int(r.last_trip_date >= active_cut_date) , axis=1)
        print(self.df.head())
        self.df['rated_driver'] = self.df['avg_rating_of_driver'].apply \
                (lambda r: int(pd.isnull(r)))
        self.df['rated_rider'] = self.df['avg_rating_by_driver'].apply \
                (lambda r: int(pd.isnull(r)))
        print(self.df.head())

        # Replace Nans in ratings
        rating_mode = self.df['avg_rating_by_driver'].mode()[0]
        self.df.avg_rating_by_driver.fillna(rating_mode, inplace=True)
        rating_mode = self.df['avg_rating_of_driver'].mode()[0]
        self.df.avg_rating_of_driver.fillna(rating_mode, inplace=True)

        # Drop the dates
        self.df.drop(['signup_date', 'last_trip_date', 'active_days'], axis=1, inplace=True)

    def _dump(self):
        '''
        Dump the cleaned dataframe to a Pickle.
        '''
        print('Dumping {} pickle'.format(self.pkl_filename))
        f = open(self.pkl_filename, 'wb')
        pickle.dump(self.df, f)
        f.close()

        print('Writing {} CSV'.format(self.out_csv_filename))
        self.df.to_csv(self.out_csv_filename)


if __name__ == '__main__':
    dm = DataPrepMgr('data/churn_train', split=False, standardize=False, save=False)
    df = dm.df
