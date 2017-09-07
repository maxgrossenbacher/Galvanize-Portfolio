import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataPrep import DataPrepMgr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score

def grid_search(model, parameters, X, y):
    clf = GridSearchCV(model, parameters)
    clf.fit(X, y)
    return clf

def forest(X, y, trees,max_features, model):
    rf = model(n_estimators=trees, max_features=max_features, n_jobs=-1)
    rf.fit(X,y)
    return rf


if __name__ == '__main__':
    dm = DataPrepMgr('churn_train', split=True, standardize=True)
    df = dm.df
    print(df.info())

    y_train = dm.y_train
    X_train = dm.X_train
    X_test = dm.X_test
    y_test = dm.y_test

    '''
    Grid Search on random forest model
    '''
    # rand_for = RandomForestClassifier(n_jobs=-1)
    # parameters ={'n_estimators':(10,100,500, 1000, 1500, 2000), 'max_features': (2,3,4,5,6,7,8,9)}
    # clf = grid_search(rand_for, parameters, X_train, y_train)


    '''
    Random Forest and scoring model
    '''
    # rf = forest(X_train, y_train, 1000, 7, RandomForestClassifier)
    # fi = rf.feature_importances_
    # print('RandomForestClassifier----------')
    # print('important feauters mat:\n',fi)
    # sort = fi.argsort()
    # cols = list(no_dates.columns)
    # feature_import_order = []
    # for ind in sort[::-1]:
    #     feature_import_order.append(cols[ind])
    # print ('important features:\n', (feature_import_order))
    #
    # print('acc score:\n',(rf.score(X_test, y_test)))
    # pred = rf.predict(X_test)
    # print('precision score:\n', (precision_score(y_test, pred)))
    # print('recall score:\n', (recall_score(y_test, pred)))
    # print('--------------------------------------\n')


    '''
    Gradient Forest and scoring model
    '''
    # n_estimators=1000, max_features=7, learning_rate=0.1
    gd = GradientBoostingClassifier()
    rf = gd.fit(X_train, y_train)
    fi = rf.feature_importances_
    print('GradientBoostingClassifier----------')
    print('important feauters mat:\n',fi)
    sort = fi.argsort()
    df_test = dm.X_test
    cols = list(df_test.columns)
    print(cols)
    feature_import_order = []
    for ind in sort[::-1]:
        feature_import_order.append(cols[ind])
    print ('important features:\n', (feature_import_order))

    print('acc score:\n',(rf.score(X_test, y_test)))
    pred = rf.predict(X_test)
    print('precision score:\n', (precision_score(y_test, pred)))
    print('recall score:\n', (recall_score(y_test, pred)))
    print('--------------------------------------\n')

    '''
    GridSearchCV GradientBoostingClassifier
    '''
    # gd = GradientBoostingClassifier()
    # clf = grid_search
    # parameters ={'n_estimators':(1000, 1500), 'max_features': (7,8,9,10), 'learning_rate':(0.01,0.1,1), 'max_depth':(2,3,4,5), 'min_samples_split':(2,3)}
    # clf = grid_search(gd, parameters, X_train, y_train)

    '''
    AdaBoost Forest and scoring model
    '''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    # ada = AdaBoostClassifier(n_estimators=1000, learning_rate=1.0)
    # rf = ada.fit(X_train, y_train)
    # fi = rf.feature_importances_
    # print('AdaBoostingClassifier----------')
    # print('important feauters mat:\n',fi)
    # sort = fi.argsort()
    # cols = list(no_dates.columns)
    # feature_import_order = []
    # for ind in sort[::-1]:
    #     feature_import_order.append(cols[ind])
    # print ('important features:\n', (feature_import_order))
    #
    # print('acc score:\n',(rf.score(X_test, y_test)))
    # pred = rf.predict(X_test)
    # print('precision score:\n', (precision_score(y_test, pred)))
    # print('recall score:\n', (recall_score(y_test, pred)))
    # print('--------------------------------------\n')
