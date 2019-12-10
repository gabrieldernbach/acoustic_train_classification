import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from utils import normalize_data, treshhold_labels


def load_data(datapath):
    if os.path.exists(datapath):
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))
    else:
        os.system('data_monolithic_mfcc_py')
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))
    return train, dev, test


def eval_model(clf):
    clf.fit(X_train, Y_train)
    print('=== Training Set Performance ===')
    print(clf.score(X_train, Y_train))
    print(confusion_matrix(Y_train, clf.predict(X_train)))
    print(roc_auc_score(Y_train, clf.predict_proba(X_train)[:, 1]))
    print('=== Dev Set Performance ===')
    print(clf.score(X_dev, Y_dev))
    print(confusion_matrix(Y_dev, clf.predict(X_dev)))
    print(roc_auc_score(Y_dev, clf.predict_proba(X_dev)[:, 1]))
    print('=== Test Set Performance')
    print(clf.score(X_test, Y_test))
    print(confusion_matrix(Y_test, clf.predict(X_test)))
    print(roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1]))


clf = {
    'RF': RandomForestClassifier(n_estimators=400,
                                 max_depth=10,
                                 n_jobs=-1,
                                 verbose=True,
                                 oob_score=True,
                                 ),

    'MLP': MLPClassifier(hidden_layer_sizes=(50, 20),
                         learning_rate_init=0.01, max_iter=400,
                         random_state=1, verbose=True, early_stopping=True)
}

if __name__ == '__main__':
    train, dev, test = load_data('data_monolithic_mfcc.pkl')

    X_train, S_train, Y_train = train
    X_dev, S_dev, Y_dev = dev
    X_test, S_test, Y_test = test
    X_train, X_dev, X_test = normalize_data(X_train, X_dev, X_test)
    Y_train, Y_dev, Y_test = treshhold_labels(Y_train, Y_dev, Y_test, .25)

    eval_model(clf['RF'])

    eval_model(clf['MLP'])
