import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from data_prep import data_processing as dp


def log_reg(X_train, X_test, y_train, y_test, col_names, print_coef=False):
    lr = LogisticRegression(fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=1, max_iter=1000)
    model = lr.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if print_coef:
        print('\nCoeficie=nts:')
        idx = np.argsort(abs(model.coef_[0])*-1)
        coefs = model.coef_[0]
        features = np.array(col_names)[idx]
        for i in idx:
            print('{:22s}: {:6.3f}'.format(features[i], coefs[i]))
    print('Recall Score (train): ', recall_score(y_train, y_pred_train))
    print('Recall Score (test): ', recall_score(y_test,y_pred_test))
    return


if __name__ == '__main__':
    filename = '../data/'+'2000sample_alltextdatesdropped.csv'
    print('\nLogistic Regression with StandardScaler=False')
    X_train, X_test, y_train, y_test, col_names = dp(filename, scale=False )
    log_reg(X_train, X_test, y_train, y_test, col_names)
    # print('\nLogistic Regression with StandardScaler=True')
    # X_train, X_test, y_train, y_test, col_names = dp(filename, scale=True )
    # log_reg(X_train, X_test, y_train, y_test, col_names)
