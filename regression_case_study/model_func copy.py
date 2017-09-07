
def predict(X_train,y_train):
    model = MultiTaskLassoCV(n_jobs=-1)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    return pred


def score(y_test,pred):
    log_diff = np.log(pred + 1) - np.log(y_test + 1)
    return np.sqrt(np.mean(log_diff**2))
