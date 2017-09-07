import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DataPrep import DataPrepMgr



#variance explained ratio and number of components
def scree_plt(X):
    #fit PCA model
    pca = PCA()
    model = pca.fit(X)
    variance_pct = model.explained_variance_ratio_
    num = pca.n_components_

    fig, ax = plt.subplots(figsize=(13,13))
    ax.bar(np.arange(num), variance_pct)
    ax.set_xlabel('Number Components')
    ax.set_ylabel('Variance Expalined Ratio')
    ax.set_title('Scree Plot')
    plt.rcParams.update({'font.size': 39})
    ax.legend()
    plt.savefig('scree_plot.png')

def decompostion(model, X, num_components):
    pca = model(n_components= num_components)
    nmf = pca.fit(X)
    trans_mat = pca.fit_transform(X)
    print(trans_mat.shape)
    return trans_mat, nmf

if __name__ == '__main__':
    dm = DataPrepMgr('/Users/gmgtex/Desktop/churn_train', split=True, standardize=True)
    df = dm.df

    y_train = dm.y_train
    X_train = dm.X_train
    X_test = dm.X_test
    y_test = dm.y_test

    scree_plt(X_train)
    plt.show()
    num_components = 2
    trans_mat, pca = decompostion(NMF, abs(X_train), num_components)
