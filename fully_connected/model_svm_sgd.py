"""
Experiment on expanding the original RBF SVM.
Experiments turned out bad classifier
"""

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from fully_connected.utils import treshhold_labels, normalize_data, load_monolithic

if __name__ == '__main__':
    train, dev, test = load_monolithic('data_monolithic_mfcc.pkl')

    X_train, S_train, Y_train = train
    X_dev, S_dev, Y_dev = dev
    X_test, S_test, Y_test = test
    X_train, X_dev, X_test = normalize_data(X_train, X_dev, X_test)
    Y_train, Y_dev, Y_test = treshhold_labels(Y_train, Y_dev, Y_test, .25)

    # rbf_feature = RBFSampler(gamma=1, n_components=800, random_state=1)
    rbf_feature = Nystroem(gamma=1, n_components=200, random_state=1)
    print('transform features')
    X_train_features = rbf_feature.fit_transform(X_train)
    X_dev_features = rbf_feature.transform(X_dev)
    print('finish')
    clf = SGDClassifier(max_iter=400, loss='log', n_jobs=-1, random_state=1,
                        alpha=0.00000001, tol=1e-9, early_stopping=False,
                        verbose=1, n_iter_no_change=40)

    clf.fit(X_train_features, Y_train)
    print('=== Training Set Performance ===')
    print(clf.score(X_train_features, Y_train))
    print(confusion_matrix(Y_train, clf.predict(X_train_features)))
    print(roc_auc_score(Y_train, clf.predict_proba(X_train_features)[:, 1]))
    print('=== Dev Set Performance ===')
    print(clf.score(X_dev_features, Y_dev))
    print(confusion_matrix(Y_dev, clf.predict(X_dev_features)))
    print(roc_auc_score(Y_dev, clf.predict_proba(X_dev_features)[:, 1]))
