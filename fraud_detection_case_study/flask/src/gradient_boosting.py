from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import data_prep as data_prep
from sklearn.metrics import recall_score, precision_score

def gradient_boosted_trees(X_train, X_test, y_train, y_test, cols):
    grad_b = GradientBoostingClassifier(loss='deviance', \
                                        learning_rate=0.1, \
                                        n_estimators=100, \
                                        subsample=1.0, \
                                        criterion='friedman_mse', \
                                        min_samples_split=2, \
                                        min_samples_leaf=1, \
                                        min_weight_fraction_leaf=0.0, \
                                        max_depth=3, \
                                        min_impurity_decrease=0.0, \
                                        min_impurity_split=None, \
                                        init=None, \
                                        random_state=1, \
                                        max_features=None, \
                                        verbose=0, \
                                        max_leaf_nodes=None, \
                                        warm_start=False, \
                                        presort='auto')
    grad_b.fit(X_train, y_train)
    predictions = grad_b.predict(X_test)
    probabilities = grad_b.predict_proba(X_test)
    return grad_b, predictions, probabilities

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cols = data_prep.data_processing('../data/2000sample_alltextdatesdropped.csv', scale=True)
    
    grad_model, predictions, probabilities = gradient_boosted_trees(X_train, X_test, y_train, y_test, cols)
    print('------------scoring model ------------------')
    print('recall score: {}'.format(recall_score(y_test, predictions)))
    print('accuracy: {}'.format(grad_model.score(X_test, y_test)))
    print('precision score: {}'.format(precision_score(y_test, predictions)))
    print('------------probability of fraud----------------')
    print('proability of fraud: {}'.format(probabilities))
    print('------------important features ---------------')
    # print('feature_importances: {}, len{}'.format(rf_model.feature_importances_, \
                                                    # len(rf_model.feature_importances_)))
    important_feats = {}
    for i, importance in enumerate(grad_model.feature_importances_):
        important_feats[cols[i]] = importance
    sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    print(sorted_feats)
