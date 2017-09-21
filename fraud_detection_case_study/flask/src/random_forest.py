import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import data_prep as data_prep
from sklearn.metrics import recall_score, precision_score

X_train, X_test, y_train, y_test, cols = data_prep.data_processing('../data/2000sample_alltextdatesdropped.csv', scale=True)

print('ratio fraud to total:',sum(y_test)/len(y_test))
rf = RandomForestClassifier(n_estimators=100, criterion='gini', \
                            max_depth=None, min_samples_split=2, \
                            min_samples_leaf=1, \
                            min_weight_fraction_leaf=0.0, \
                            max_features='auto', max_leaf_nodes=None, \
                            min_impurity_decrease=0.0, min_impurity_split=None, \
                            bootstrap=True, oob_score=False, \
                            n_jobs=1, random_state=1, verbose=0, \
                            warm_start=False, class_weight=None)

rf_model = rf.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
probabilities = rf_model.predict_proba(X_test)


print('recall score: {}'.format(recall_score(y_test, predictions)))
print('accuracy: {}'.format(rf_model.score(X_test, y_test)))
print('precision score: {}'.format(precision_score(y_test, predictions)))
print('proability of fraud: {}'.format(probabilities))
print('feature_importances: {}, len{}'.format(rf_model.feature_importances_, \
                                                len(rf_model.feature_importances_)))

important_feats = {}
for i, importance in enumerate(rf_model.feature_importances_):
    important_feats[cols[i]] = importance
sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
print(sorted_feats)
