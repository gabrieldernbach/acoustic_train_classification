from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from librosa.feature import mfcc
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
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


for _ in range(1000):
    for dataset_name, root in datasets.items():
        register = build_register(root)
        train, test = group_split(register, random_state=np.random.randint(0, 1000), group='file_name')
        X_train, G_train, Y_train = load(train)
        X_test, G_test, Y_test = load(test)

        classifier = Pipeline([
            ('scale', StandardScaler()),
            ('quantile', QuantileTransformer(output_distribution='uniform')),
            ('sgd', SGDClassifier(loss='log', alpha=0.0001, n_jobs=1, verbose=0, tol=1e-5, max_iter=2000,
                                  early_stopping=True, n_iter_no_change=10, validation_fraction=0.2))
        ])

        parameter_distribution = {
            'quantile': [
                'passthrough',
                # QuantileTransformer(output_distribution='uniform'),
                # QuantileTransformer(output_distribution='normal')
            ],
            'sgd__loss':
                ['log', 'modified_huber'],
            # ['hinge', 'log', 'perceptron', 'modified_huber'],
            'sgd__alpha':
                loguniform(1e-7, 1e-1),
            'sgd__max_iter':
                sps.randint(400, 10_000),
            'sgd__epsilon':
                sps.uniform(0.001, 0.2),
            'sgd__class_weight':
                [
                    'balanced',
                    # None
                ]
        }

        random_search = RandomizedSearchCV(classifier,
                                           param_distributions=parameter_distribution,
                                           n_iter=40, scoring='f1',
                                           cv=GroupKFold(n_splits=3),
                                           verbose=2, n_jobs=4, pre_dispatch=4)

        random_search.fit(X_train, Y_train, groups=G_train)

        # save detailed random search info
        result_path = '/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/sgd/'
        Path(result_path).mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame(random_search.cv_results_)
        results['dataset_name'] = dataset_name
        results.to_csv(f'{result_path}exp_{datetime.now().strftime("%Y%m%dT%H%M%S")}.csv')

        # evaluate on test set and save separately
        eval_test = {
            'f1': f1_score(Y_test, random_search.best_estimator_.predict(X_test)),
            'confmat': confusion_matrix(Y_test, random_search.best_estimator_.predict(X_test)),
            'dataset': dataset_name,
        }
        eval_test = {**eval_test, **random_search.best_params_}
        pd.DataFrame([eval_test]).to_csv(f'{result_path}test_{datetime.now().strftime("%Y%m%dT%H%M%S")}.csv')
