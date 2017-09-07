
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


#Reading in data
def read_csv():
    data = pd.read_csv('data/train.csv')
    datalinear = data.copy()
    return data, datalinear

#Dropping null values in Machine Hours Current Meter
def clean_data(df, features):
    datalinear = df[['SalePrice', features]]
    datalinear = datalinear.dropna(axis=0)
    return datalinear

#creating numpy array, y-salesprice, X-features=1-MachineHoursCurrentMeter
def prepare_data(df):
    y = df.pop('SalePrice').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

#plot
def plot():
    fig, ax = plt.subplots()
    ax.scatter(x=X_train, y=y_train)
    ax.set_xlabel('MachineHoursCurrentMeter')
    ax.set_ylabel('price')

def linear_regression(X,y,X_tst, y_tst):
    simple_linear = LinearRegression(fit_intercept=True, n_jobs=-1)
    simple_linear.fit(X, y)
    y_predicted = simple_linear.predict(X_tst)
    r2 = simple_linear.score(X_tst, y_tst)
    print ('r2: {}'.format(r2))
    return r2, y_predicted

# def two_feature_linear_regression_Kfold(df, features):
#     df = df[['SalePrice', features[0], features[1]]]
#     y = df.pop('SalePrice').values
#     X = df.values
#     #train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     #Cross validation Kfold
#     r2s = []
#     kf = KFold(n_splits=10, shuffle=True)
#     for train_index, test_index in kf.split(X_train):
#         X_trn, X_tst = X_train[train_index], X_train[test_index]
#         y_trn, y_tst = y_train[train_index], y_train[test_index],
#         r2s.append(linear_regression(X_trn, y_trn, X_tst, y_tst))
#     return np.mean(r2s)

def two_feature_linear_regression(df, features):
    df = df[['SalePrice', features[0], features[1]]]
    y = df.pop('SalePrice').values
    X = df.values
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    r2, y_predicted = linear_regression(X_train,y_train, X_test, y_test)
    return y_predicted, y_test

def score(predictions, y_test):
    log_diff = np.log(predictions+1) - np.log(y_test+1)
    return np.sqrt(np.mean(log_diff**2))

if __name__ == '__main__':
    data, datalinear = read_csv()

    # print('\n--------------------------------------------')
    # print('MachineHoursCurrentMeter')
    # dataMachineH = clean_data(datalinear, 'MachineHoursCurrentMeter')
    # X_train, X_test, y_train, y_test = prepare_data(dataMachineH)
    # linear_regression(X_train, y_train, X_test, y_test)
    #
    # print('\n--------------------------------------------')
    # print('datayearmade')
    # datayearmade = clean_data(datalinear, 'YearMade')
    # X_train1, X_test1, y_train1, y_test1 = prepare_data(datayearmade)
    # linear_regression(X_train1, y_train1, X_test1, y_test1)
    #
    # print('\n--------------------------------------------')
    # print('auctioneerID')
    # data_auctioneer = clean_data(datalinear, 'auctioneerID')
    # X_train2, X_test2, y_train2, y_test2 = prepare_data(data_auctioneer)
    # linear_regression(X_train2, y_train2, X_test2, y_test2)
    #
    # print('\n--------------------------------------------')
    # print('datasource')
    # data_source = clean_data(datalinear, 'datasource')
    # X_train3, X_test3, y_train3, y_test3 = prepare_data(data_source)
    # linear_regression(X_train3, y_train3, X_test3, y_test3)
    #
    #
    # print('\n--------------------------------------------')
    # print('ModelID')
    # data_model = clean_data(datalinear, 'ModelID')
    # X_train4, X_test4, y_train4, y_test4 = prepare_data(data_model)
    # linear_regression(X_train4, y_train4, X_test4, y_test4)
    #
    # print('\n--------------------------------------------')
    # print('MachineID')
    # data_machineid = clean_data(datalinear, 'MachineID')
    # X_train5, X_test5, y_train5, y_test5 = prepare_data(data_machineid)
    # linear_regression(X_train5, y_train5, X_test5, y_test5)
    #
    # print('\n--------------------------------------------')
    # print('SalesID')
    # data_salesID = clean_data(datalinear, 'SalesID')
    # X_train6, X_test6, y_train6, y_test6 = prepare_data(data_salesID)
    # linear_regression(X_train6, y_train6, X_test6, y_test6)

    # print('\n--------------------------------------------')
    # print('MachineID & YearMade')
    # print(two_feature_linear_regression_Kfold(datalinear, features=['YearMade', 'MachineID']))


    # 
    # print('\n--------------------------------------------')
    # print('MachineID & YearMade')
    # y_predicted, y_test = two_feature_linear_regression(datalinear, features=['YearMade', 'MachineID'])
    #
    # score = score(y_predicted, y_test)
    # print('log diff score: {}'.format(score))

    #np.savetxt(simple_linear_predictions_MachineID_YearMade.csv, y_predicted)
    #plot()
    #plt.show()

    data, datalinear = read_csv()
    cols = ['MachineID', 'MachineHoursCurrentMeter', 'UsageBand','SalePrice','Enclosure','Transmission']
    data_features = data[cols]
