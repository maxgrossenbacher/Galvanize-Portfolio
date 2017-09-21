import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.feature_selection import RFE

def lin_SVC(X_train, X_test, y_train, y_test, print_coef=False):
    clf = LinearSVC(class_weight='balanced', random_state=1)
    model = clf.fit(X_train, y_train)
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
    return model

if __name__ == '__main__':
    filename = '../data/'+'2000sample_alltextdatesdropped.csv'
    print('\nLinearSVC with StandardScaler=False')
    npz = np.load('../data/Xycompressed.npz')
    X_train = npz['X_train']
    X_test = npz['X_test']
    y_train = npz['y_train']
    y_test = npz['y_test']
    svdmodel=lin_SVC(X_train, X_test, y_train, y_test)

    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(svdmodel, f)
    print('Pickled.')
