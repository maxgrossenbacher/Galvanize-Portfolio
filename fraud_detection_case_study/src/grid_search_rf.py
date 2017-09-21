import pandas as pd
import numpy as np
import data_prep as data_prep
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, make_scorer
import pickle
from sklearn.model_selection import GridSearchCV

def grid_search(model, X_train, y_train, params, scoring, cv):
    print('running grid search...')
    grid_cv = GridSearchCV(model, params, n_jobs=-1, scoring=scoring, cv=cv)
    print('fitting grid search...')
    grid_model = grid_cv.fit(X_train, y_train)
    print('Done.')
    cv_results = pd.DataFrame(grid_model.cv_results_)
    return grid_model, cv_results


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cols = data_prep.data_processing('../data/2000sample_alltextdatesdropped.csv', scale=False)

    grad_b = GradientBoostingClassifier()
    params_grid = {'n_estimators':[1000, 250, 500, 750], \
                    'learning_rate':[0.5, 0.3, 0.1], \
                    'max_depth':[3, 5, 7], \
                    'max_features':[None, 5, 10]}
    print('-------------Grid Search-----------------')
    grid_gb_model, cv_results = grid_search(grad_b, X_train, y_train, params_grid, cv=5, scoring='recall')
    print('scorer {} has best_score:{} '.format (grid_gb_model.scorer_, grid_gb_model.best_score_))
    print('best_params: ', grid_gb_model.best_params_)
    print('best_estimator: ', grid_gb_model.best_estimator_)

    with open('model.pkl', 'wb') as f:
        pickle.dump(grid_gb_model, f)
    print('Pickled.')

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
