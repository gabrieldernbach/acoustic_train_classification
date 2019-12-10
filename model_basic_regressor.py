import os
import pickle

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor


def evaluate_clf():
    clf.fit(X_train, Y_train)

    print('=== Training Set Performance ===')
    print(clf.score(X_train, Y_train))
    Y_train_binary = Y_train > .25
    train_predict = clf.predict(X_train)
    print('confusion\n', confusion_matrix(Y_train_binary, train_predict > .25))
    print('roc auc:', roc_auc_score(Y_train_binary, train_predict))
    print('=== Dev Set Performance ===')
    print(clf.score(X_dev, Y_dev))
    Y_dev_binary = Y_dev > .25
    dev_predict = clf.predict(X_dev)
    print('confusion\n', confusion_matrix(Y_dev_binary, dev_predict > .25))
    print('roc auc:', roc_auc_score(Y_dev_binary, dev_predict))
    print('=== Test Set Performance')
    print(clf.score(X_test, Y_test))
    Y_test_binary = Y_test > .25
    test_predict = clf.predict(X_test)
    print('confusion:\n', confusion_matrix(Y_test_binary, test_predict > .25))
    print('roc auc:', roc_auc_score(Y_test_binary, test_predict))


if __name__ == '__main__':
    datapath = 'data_monolithic_mfcc.pkl'
    if os.path.exists(datapath):
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))
    else:
        os.system('data_monolithic_mfcc_py')
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))

    X_train, S_train, Y_train = train
    X_dev, S_dev, Y_dev = dev
    X_test, S_test, Y_test = test

    clf = HistGradientBoostingRegressor(loss='least_squares',
                                        max_iter=400, validation_fraction=0.1,
                                        verbose=1)
    print('\n Gradient Boosted Tree \n')
    evaluate_clf()

    # noramlize data
    Xmean = X_train.mean(axis=0, keepdims=1)
    Xvar = (X_train - Xmean).var(axis=0, keepdims=1)

    X_train = (X_train - Xmean) / Xvar
    X_dev = (X_dev - Xmean) / Xvar
    X_test = (X_test - Xmean) / Xvar

    clf = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20),
                       learning_rate_init=0.001, max_iter=400,
                       random_state=1, verbose=True, early_stopping=True)
    print('\n MLPRegressor with normalization \n')
    evaluate_clf()
