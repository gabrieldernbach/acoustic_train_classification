from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from librosa.feature import mfcc
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from tqdm import tqdm

from convolution_net.load import build_register, group_split

datasets = {
    'trainspeed_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_2sec',
    'trainpseed_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_5sec',
    'subsample_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_2sec',
    'subsample_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_5sec',
    'beatfrequency_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_2sec',
    'beatfrequency_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_5sec',
}


def load(register):
    flatmfcc = lambda x: mfcc(x).flatten()
    flatpool = lambda x: np.mean(x) > 0.05

    X = np.array([flatmfcc(np.load(p)) for p in tqdm(register.audio_path.values)])
    G = np.array([g for g in tqdm(register.file_id)])
    Y = np.array([flatpool(np.load(p)) for p in tqdm(register.target_path.values)])
    return X, G, Y


if __name__ == "__main__":
    for _ in range(1000):
        for dataset_name, root in datasets.items():
            register = build_register(root)
            train, test = group_split(register, random_state=np.random.randint(0, 1000), group='file_name')
            X_train, G_train, Y_train = load(train)
            X_test, G_test, Y_test = load(test)

            classifier = Pipeline([
                ('scale', StandardScaler()),
                ('quantile', QuantileTransformer(output_distribution='uniform')),
                ('svc', SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced',
                            verbose=0, tol=1e-3, cache_size=1000))
            ])

            parameter_distribution = {
                'quantile': [
                    'passthrough',
                    QuantileTransformer(output_distribution='uniform'),
                    # QuantileTransformer(output_distribution='normal')
                ],
                'svc__C':
                    sps.uniform(0.1, 2.0),
                'svc__gamma':
                    loguniform(1e-7, 1e-1),
                'svc__class_weight': [
                    'balanced',
                    # None
                ],
            }
            random_search = RandomizedSearchCV(estimator=classifier,
                                               param_distributions=parameter_distribution,
                                               n_iter=40, scoring='f1',
                                               cv=GroupKFold(n_splits=3),
                                               verbose=2, n_jobs=4, pre_dispatch=8)

            random_search.fit(X_train, Y_train, groups=G_train)

            # save detailed random search info
            result_path = '/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/svm/'
            Path(result_path).mkdir(parents=True, exist_ok=True)
            results = pd.DataFrame(random_search.cv_results_)
            results['dataset_name'] = dataset_name
            results.to_csv(f'{result_path}exp_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.csv')

            # evaluate on test set and save separately
            eval_test = {
                'f1': f1_score(Y_test, random_search.best_estimator_.predict(X_test)),
                'confmat': confusion_matrix(Y_test, random_search.best_estimator_.predict(X_test)),
                'dataset': dataset_name,
            }
            eval_test = {**eval_test, **random_search.best_params_}
            pd.DataFrame([eval_test]).to_csv(f'{result_path}test_{datetime.now().strftime("%Y-%m-%d-%-H:%M:%S")}.csv')

            # clf = HistGradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=40, verbose=0, max_iter=200)
            #
            # param_dist = {
            #     'learning_rate': sps.uniform(0.1, 0.9),
            #     'max_iter': sps.randint(50, 500),
            #     'max_leaf_nodes': sps.randint(2, 80),
            #     'max_depth': sps.randint(2, 100),
            #     'min_samples_leaf': sps.randint(2, 60),
            #     'l2_regularization': sps.uniform(0.0, 1.0),
            #     'validation_fraction': sps.uniform(0.2, 0.4),
            #     'n_iter_no_change': sps.randint(40, 200),
            # }
