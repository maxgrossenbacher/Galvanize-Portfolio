import pandas as pd
import numpy as np
import data_prep as data_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV

def grid_search(model, X_train, y_train, params, scoring, cv):
    print('running grid search...')
    grid_cv = GridSearchCV(rf, params, n_jobs=-1)
    print('fitting grid search...')
    grid_model = grid_cv.fit(X_train, y_train)
    print('Done.')
    cv_results = pd.DataFrame(grid_model.cv_results_)
    return grid_model, cv_results


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cols = data_prep.data_processing('../data/2000sample_alltextdatesdropped.csv', scale=False)

    rf = RandomForestClassifier(random_state=1)
    params_grid = {'n_estimators':[1000, 250, 500, 750], \
                    'criterion':['gini'], \
                    'max_features':['auto', 'log2', 5, 10, 15], \
                    'max_depth':[5, 10, 15], \
                    'min_samples_split':[0.1, 0.05], \
                    'min_impurity_decrease':[0.01, 0.1, 0.001]}
    print('-------------Grid Search-----------------')
    grid_rf_model, cv_results = grid_search(rf, X_train, y_train, params_grid, cv=5, scoring='recall')
    print('scorer {} has best_score:{} '.format (grid_rf_model.scorer_, grid_rf_model.best_score_))
    print('best_params: ', grid_rf_model.best_params_)
    print('best_estimator: ', grid_rf_model.best_estimator_)

'''
best_score:  0.930425378515
best_params:  {'criterion': 'gini', 'max_depth': 15, 'max_features': 10, 'min_impurity_decrease': 0.001, 'min_samples_split': 0.05, 'n_estimators': 1000}
best_estimator:  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=0.001, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=0.05,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

'''
