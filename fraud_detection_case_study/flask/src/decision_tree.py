import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import data_prep as data_prep
from sklearn.metrics import recall_score, precision_score
import graphviz

def decision_tree(X_train, X_test, y_train, y_test):
    trees = DecisionTreeClassifier()
    tree_model = trees.fit(X_train, y_train)
    predictions = tree_model.predict(X_test)
    probabilities = tree_model.predict_proba(X_test)
    return tree_model, predictions, probabilities


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cols = data_prep.data_processing('../data/2000sample_alltextdatesdropped.csv', scale=True)
    tree_model, predictions, probabilities = decision_tree(X_train, X_test, y_train, y_test)

    print('recall score: {}'.format(recall_score(y_test, predictions)))
    print('accuracy: {}'.format(tree_model.score(X_test, y_test)))
    print('precision score: {}'.format(precision_score(y_test, predictions)))
    print('proability of fraud: {}'.format(probabilities))
    print('feature_importances: {}, len{}'.format(tree_model.feature_importances_, len(tree_model.feature_importances_)))
    important_feats = {}
    for i, importance in enumerate(tree_model.feature_importances_):
        important_feats[cols[i]] = importance
    sorted_feats = sorted(important_feats.items(), key=lambda x: x[1])[::-1]
    print(sorted_feats)


    # dot_data = tree.export_graphviz(tree_model, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("Fraud")
    #
    # dot_data = tree.export_graphviz(tree_model, out_file=None,
    #                          feature_names=cols,
    #                          class_names=y,
    #                          filled=True, rounded=True,
    #                          special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph
