import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

def col_drop(df_name, col):
    df_name = df_name.drop(col, axis = 1)
    return df_name

def feat_targ(df):
    y = df['target'].values
    Features = col_drop(df, ['Unnamed: 0', 'target'])
    return (Features, y)

def train_test(Features, y):
    X = Features.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
    return(X_train, X_test, y_train, y_test)

def log_reg(Features, X_train, X_test, y_train, y_test):
    lr = LogisticRegression(fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=1, max_iter=1000)
    model = lr.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print('\n \nLogistic Regression Recall Score (train): ', recall_score(y_train, y_pred_train))
    # print(y_pred_train, y_train)
    print('Logistic Regression Recall Score (test): ', recall_score(y_test,y_pred_test))
    print('Logistic Regression Coeficients:')
    idx = np.argsort(abs(model.coef_[0])*-1)
    coefs = model.coef_[0]
    features = np.array(Features.columns)[idx]
    for i in idx:
        print('{:22s}: {:6.3f}'.format(features[i], coefs[i]))
    return(features, coefs)

# def remove_features(Features, thres, features, coefs):
#     for i in range(len(features)):
#         # print('{:22s}: {:6.3f}'.format(features[i], coefs[i]))
#         # print(thres, coefs.max())
#         if abs(coefs[i]) < thres*abs(coefs.max()):
#             Features = col_drop(Features, features[i])
#     return(Features, len(features), abs(coefs).max(), abs(coefs).min())


if __name__ == '__main__':
    # filename = input('Enter the data file:\n')
    filename = '2000sample_alltextdatesdropped.csv'
    df = pd.read_csv('../data/'+filename)
    Features, y = feat_targ(df)
    X_train, X_test, y_train, y_test = train_test(Features, y)
    features, coefs = log_reg(Features, X_train, X_test, y_train, y_test)
