import numpy as np
import pandas as pd
import data_preprocessing as dp
from sklearn.linear_model import LassoCV
from sklearn.model_selection import  train_test_split

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def model(X_train,y_train, X_test):
    model = LassoCV(n_alphas=400, n_jobs=-1)
    return model.fit(X_train,y_train)

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred


def score(y_test,pred):
    log_diff = np.log(pred + 1) - np.log(y_test + 1)
    return np.sqrt(np.mean(log_diff**2))


if __name__ == '__main__':

    #Read in training data
    df = pd.read_csv('data/Train.csv')

    #

    corr = ['SalePrice','Enclosure','ProductGroup','Hydraulics_Flow','ProductSize','Travel_Controls']
    corrmodel = df[corr]
    test = pd.get_dummies(corrmodel,dummy_na=True, drop_first=True)
    y = test.pop('SalePrice').values

    # bring in test data
    dft = pd.read_csv('data/test.csv')
    zz = dft['SalesID'].values
    corrt = ['Enclosure','ProductGroup','Hydraulics_Flow','ProductSize','Travel_Controls']
    corrmodelt = dft[corrt]
    dft = pd.get_dummies(corrmodelt,dummy_na=True, drop_first=True)
    cols = list(dft.columns)

    # only use cols from training
    test = test[cols]
    X = test.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = model(X_train, y_train, X_test)
    y_pred = model.predict(X_test)
    score = score(y_test, y_pred)

    XX = dft.values
    dft = pd.read_csv('data/test.csv')
    zz = dft['SalesID'].values
    corrt = ['Enclosure','ProductGroup','Hydraulics_Flow','ProductSize','Travel_Controls']
    corrmodelt = dft[corrt]
    dft = pd.get_dummies(corrmodelt,dummy_na=True, drop_first=True)
    XX = dft.values
    yy = model.predict(XX)

    results = np.c_[zz,yy]
    # np.savetxt('predictions_lassocv.csv', results, delimiter=',')
    results_df = pd.DataFrame(data=results,columns=['SalesID','SalePrice'])
    # results_df.to_csv('our_predictions.csv')
