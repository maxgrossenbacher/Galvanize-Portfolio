import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve
import graphviz
import pickle as pickle

def trees_model(model, X_train, X_test, y_train, y_test):
    tree_model = model.fit(X_train, y_train)
    predictions = tree_model.predict(X_test)
    probabilities = tree_model.predict_proba(X_test)
    return tree_model, predictions, probabilities


if __name__ == '__main__':
    npz = np.load('../data/Xycompressed.npz')
    X_train = npz['X_train']
    X_test = npz['X_test']
    y_train = npz['y_train']
    y_test = npz['y_test']
    # print('\n --> Decision Tree------------------')
    # dt = DecisionTreeClassifier(criterion='gini', \
    #                             splitter='best', \
    #                             max_depth=None, \
    #                             min_samples_split=2, \
    #                             min_samples_leaf=1, \
    #                             min_weight_fraction_leaf=0.0, \
    #                             max_features=None, \
    #                             random_state=1, \
    #                             max_leaf_nodes=None, \
    #                             min_impurity_decrease=0.0, \
    #                             min_impurity_split=None, \
    #                             class_weight=None, \
    #                             presort=False)
    # dt_model, predictions, probabilities = trees_model(dt,X_train, X_test, y_train, y_test)
    # print('recall score: {}'.format(recall_score(y_test, predictions)))
    # print('accuracy: {}'.format(dt_model.score(X_test, y_test)))
    # print('precision score: {}'.format(precision_score(y_test, predictions)))
    # print('------------important features DT---------------')
    # # print('feature_importances: {}, len{}'.format(tree_model.feature_importances_, len(tree_model.feature_importances_)))
    # important_feats = {}
    # for i, importance in enumerate(dt_model.feature_importances_):
    #     important_feats[cols[i]] = importance
    # sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    # print(sorted_feats)


    # print('\n --> Random Forest------------------')
    # rf = RandomForestClassifier(n_estimators=1000, criterion='gini', \
    #                             max_depth=None, min_samples_split=2, \
    #                             min_samples_leaf=1, \
    #                             min_weight_fraction_leaf=0.0, \
    #                             max_features=7, max_leaf_nodes=None, \
    #                             min_impurity_decrease=0.0, min_impurity_split=None, \
    #                             bootstrap=True, oob_score=False, \
    #                             n_jobs=-1, random_state=1, verbose=0, \
    #                             warm_start=False, class_weight=None)
    # rf_model, predictions, probabilities = trees_model(rf,X_train, X_test, y_train, y_test)
    # print('------------scoring model RF------------------')
    # print('recall score: {}'.format(recall_score(y_test, predictions)))
    # print('accuracy: {}'.format(rf_model.score(X_test, y_test)))
    # print('precision score: {}'.format(precision_score(y_test, predictions)))
    # print('under_ROC_score: {}'.format(roc_auc_score(y_test, probabilities[:,1])))
    # # print('Roc curve: {}'.format(roc_curve(y_test, probabilities[:,1])))
    # print('------------probability of fraud----------------')
    # print('proability of fraud: {}'.format(probabilities))
    # print('------------important features RF---------------')
    # # print('feature_importances: {}, len{}'.format(rf_model.feature_importances_, \
    #                                                 # len(rf_model.feature_importances_)))
    # important_feats = {}
    # for i, importance in enumerate(rf_model.feature_importances_):
    #     important_feats[cols[i]] = importance
    # sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    # print(sorted_feats)

    print('\n --> Gradient Boosted Forest------------------')
    grad_b = GradientBoostingClassifier(learning_rate= 0.5, max_depth= 7, max_features= None, n_estimators= 500)#learning_rate= 0.1, max_depth= 7, max_features= 5, n_estimators= 500)
    gb_model, predictions, probabilities = trees_model(grad_b,X_train, X_test, y_train, y_test)
    print('------------scoring model GB------------------')
    print('recall score: {}'.format(recall_score(y_test, predictions)))
    print('accuracy: {}'.format(gb_model.score(X_test, y_test)))
    print('precision score: {}'.format(precision_score(y_test, predictions)))
    print('under_ROC_score: {}'.format(roc_auc_score(y_test, probabilities[:,1])))
    # print('Roc curve: {}'.format(roc_curve(y_test, probabilities[:,1])))
    print('------------probability of fraud----------------')
    print('proability of fraud: {}'.format(probabilities))
    # print('------------important features GB---------------')
    # # print('feature_importances: {}, len{}'.format(rf_model.feature_importances_, \
    #                                                 # len(rf_model.feature_importances_)))
    # important_feats = {}
    # for i, importance in enumerate(gb_model.feature_importances_):
    #     important_feats[cols[i]] = importance
    # sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    # print(sorted_feats)
    # dot_data = tree.export_graphviz(decision_tree_model, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("Fraud")
    #
    # dot_data = tree.export_graphviz(decision_tree_model, out_file=None,
    #                          feature_names=cols,
    #                          class_names=y,
    #                          filled=True, rounded=True,
    #                          special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph


    print('\n --> Random Forest------------------')
    rf = RandomForestClassifier(criterion= 'gini', max_depth= 15, max_features= 10, min_impurity_decrease= 0.001, min_samples_split= 0.05, n_estimators= 1000, class_weight='balanced')
    rf_model, predictions, probabilities = trees_model(rf,X_train, X_test, y_train, y_test)
    print('------------scoring model RF------------------')
    print('recall score: {}'.format(recall_score(y_test, predictions)))
    print('accuracy: {}'.format(rf_model.score(X_test, y_test)))
    print('precision score: {}'.format(precision_score(y_test, predictions)))
    print('under_ROC_score: {}'.format(roc_auc_score(y_test, probabilities[:,1])))
    # print('Roc curve: {}'.format(roc_curve(y_test, probabilities[:,1])))
    print('------------probability of fraud----------------')
    print('proability of fraud: \n{}'.format(probabilities))

    #pickle the model
    with open('gdmodel.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    print('Pickled.')

    # print('------------important features RF---------------')
    # # print('feature_importances: {}, len{}'.format(rf_model.feature_importances_, \
    #                                                 # len(rf_model.feature_importances_)))
    # important_feats = {}
    # for i, importance in enumerate(rf_model.feature_importances_):
    #     important_feats[cols[i]] = importance
    # sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    # print(sorted_feats)
