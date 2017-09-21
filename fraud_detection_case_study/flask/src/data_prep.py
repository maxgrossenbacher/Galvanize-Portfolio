import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import nlp_fraud


def data_processing(filepath='../data/2000sample_alltextdatesdropped.csv', scale=False):
    data = pd.read_csv('../data/2000sample_alltextdatesdropped.csv')
    filename = '../data/2000rows_withnewfeatures.json'
    df = nlp_fraud.nlp(filename, max_features=500, n_clusters=5)
    data = pd.concat([data, pd.get_dummies(df,prefix='nlp')], axis=1)
    y = data.pop('target')
    unnamed = data.pop('Unnamed: 0')
    cols = list(data.columns)
    numerical_vals = data.select_dtypes(exclude=['object', 'bool'])
    cat_vals = data.select_dtypes(include=['object', 'bool'])
    X_train, X_test, y_train, y_test = train_test_split(data, y, stratify=y, random_state=42)
    if scale:
        scalar = StandardScaler()
        scaled_X_train = scalar.fit_transform(X_train[numerical_vals.columns])
        scaled_X_test = scalar.transform(X_test[numerical_vals.columns])
        print(pd.DataFrame(scaled_X_train, columns=list(numerical_vals.columns)))
        print(cat_vals)
        scaled_X_train=pd.DataFrame(scaled_X_train)
        scaled_X_test = pd.DataFrame(scaled_X_test)
        scale_X_train = pd.concat([scaled_X_train, X_train[cat_vals]])
        scale_X_test = pd.concat([scaled_X_test, X_test[cat_vals]])
        scaled_X_train_resampled, y_train_resampled = SMOTE().fit_sample(scale_X_train, y_train)
        print(sorted(Counter(y_train_resampled).items()))
        return scaled_X_train_resampled, scale_X_test, y_train_resampled, y_test, cols
    else:
        X_train_resampled, y_train_resampled = SMOTE(k_neighbors=3, m_neighbors=5).fit_sample(X_train, y_train)
        print(sorted(Counter(y_train_resampled).items()))
        return X_train_resampled, X_test, y_train_resampled, y_test, cols
