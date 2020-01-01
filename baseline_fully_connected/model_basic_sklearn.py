from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor, MLPClassifier

from baseline_fully_connected.utils import load_monolithic


def evaluate(clf, inputs, labels):
    print(clf.score(inputs, labels))
    labels_binary = labels > .25
    prediction = clf.predict(inputs)
    print('confusion\n', confusion_matrix(labels_binary, prediction > .25))
    print('roc auc:', roc_auc_score(labels_binary, prediction))


def train_evaluate(clf, train, dev, test):
    # print(f'\n{clf.__class__}\n')
    X_train, S_train, Y_train = train
    X_dev, S_dev, Y_dev = dev
    X_test, S_test, Y_test = test

    clf.fit(X_train, Y_train)
    print('=== Training Set Performance ===')
    evaluate(clf, X_train, Y_train)
    print('=== Dev Set Performance ===')
    evaluate(clf, X_dev, Y_dev)
    print('=== Test Set Performance')
    evaluate(clf, X_test, Y_test)


clf = {

    'GBRT': HistGradientBoostingRegressor(loss='least_squares',
                                          max_iter=200, validation_fraction=0.1,
                                          verbose=True),

    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(50, 20),
                                 learning_rate_init=0.01, max_iter=400,
                                 random_state=1, verbose=False, early_stopping=True),

    'RF': RandomForestClassifier(n_estimators=200,
                                 max_depth=10,
                                 n_jobs=-1,
                                 verbose=False,
                                 oob_score=True,
                                 ),

    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(50, 20),
                                   learning_rate_init=0.01, max_iter=400,
                                   random_state=1, verbose=False, early_stopping=True)
}

if __name__ == '__main__':
    subsets = ['data_monolithic_mfcc_BHV.pkl',
               'data_monolithic_mfcc_BRL.pkl',
               'data_monolithic_mfcc_VLD.pkl',
               'data_monolithic_mfcc.pkl', ]

    for i in subsets:
        for j in subsets:
            print(f'\nlearn on {i} \npredict on {j}')
            train, dev, _ = load_monolithic(i)
            _, _, test = load_monolithic(j)

            train_evaluate(clf['GBRT'], train, dev, test)

            # compare to other classifiers
            # X_train, X_dev, X_test = normalize_data(X_train, X_dev, X_test)
            # train_evaluate(clf['MLPRegressor'])
            # Y_train, Y_dev, Y_test = treshhold_labels(Y_train, Y_dev, Y_test, .25)
            # train_evaluate(clf['RF'])
            # train_evaluate(clf['MLPClassifier'])
