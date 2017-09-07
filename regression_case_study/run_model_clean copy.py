import numpy as np
import pandas as pd
import data_preprocessing as dp
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import  train_test_split

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def find_corrlations(df, threshold):
    cols = [x for x in df.columns if df[x].value_counts().nunique() < 50]
    cols.append('SalePrice')
    indx = np.random.choice(list(range(df.shape[0])),10000)
    rand = df.loc[indx]
    rand = rand[cols]
    dftest = pd.get_dummies(rand,dummy_na=True, drop_first=True)
    x = dftest.corr()['SalePrice']
    upper = x[x>threshold].sort_values()
    lower = x[x<-threshold].sort_values()
    print ('Upper: {} \n Lower: {}'.format(upper, lower))
    return

def dummies(df, df_test, cols):
    # read in train datafram and test dataframe and cols to make dummies
    SalesID = df_test.pop('SalesID').values
    # mask by columns test and train
    corrmodel = df[cols]
    cols.remove('SalePrice')
    corrmodel_t = df_test[cols]


    #create dummy variables for train
    train = pd.get_dummies(corrmodel,dummy_na=True, drop_first=True)
    #pop sales price in to narray
    y = train.pop('SalePrice').values

    #create dummies for test sample
    dft = pd.get_dummies(corrmodel_t,dummy_na=True, drop_first=True)

    #Compare columns in test and train dataframes
    columns = list(dft.columns)
    # only use columns in both train and test dataframes
    train = train[columns]
    #create np array
    X = train.values
    predict_on_these = dft.values

    return X, y, SalesID, predict_on_these


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def lasso_model(X_train, y_train):
    model = LassoCV(n_alphas=400, n_jobs=-1)
    return model.fit(X_train,y_train)

def ridge_model(X_train, y_train):
    model = RidgeCV(alphas=[0.001,0.01,0.1,1,10])
    return model.fit(X_train,y_train)

def linear_regression(X_train, y_train):
    model = LinearRegression()
    return model.fit(X_train, y_train)

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred

def score(y_test,pred):
    log_diff = np.log(pred + 1) - np.log(y_test + 1)
    return np.sqrt(np.mean(log_diff**2))


def write_csv(test_predictions, ID):
    results = np.c_[test_predictions,ID]
    results_df = pd.DataFrame(data=results,columns=['SalesID','SalePrice'])
    results_df.to_csv('our_predictions.csv', index=False)
    return



if __name__ == '__main__':
    df = read_csv('data/Train.csv')
    df_test = read_csv('data/test.csv')

    find_corrlations(df, 0.2)

    # create dummy variables
    corr = ['SalePrice','Enclosure','ProductGroup','Hydraulics_Flow','ProductSize','Travel_Controls']
    X, y, SalesID, predict_on_these = dummies(df, df_test, corr)

    #split data into train test split
    X_train, X_test, y_train, y_test = split_data(X,y)

    #fit model to LassoCV
    lasso = lasso_model(X_train, y_train)

    # using train test split to score model
    y_predictions = predict(lasso,  X_test)
    print('lasso model:', score(y_test, y_predictions))

    #predicting on test.csv and writing csv
    test_predictions = predict(lasso, predict_on_these)
    #write_csv(test_predictions, SalesID)


    ridge = ridge_model(X_train, y_train)
    ridge_predict = predict(ridge, X_test)
    print('--------------------------------------')
    print('ridge model: ',score(y_test, ridge_predict))

    linear = linear_regression(X_train, y_train)
    linear_predict = predict(linear, X_test)
    print('------------------------------------------')
    print('linear model:',score(y_test, linear_predict))
