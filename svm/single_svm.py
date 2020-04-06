from datetime import datetime

import numpy as np
import pandas as pd
from librosa.feature import mfcc
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    for random_state in [0, 1, 2]:
        for dataset_name, root in datasets.items():
            register = build_register(root)
            train, test = group_split(register, random_state=random_state, group='file_name')
            X_train, G_train, Y_train = load(train)
            X_test, G_test, Y_test = load(test)

            classifier = Pipeline([
                ('scale', StandardScaler()),
                ('svc', SVC(C=1, kernel='rbf', gamma=0.001, class_weight='balanced'))
            ])

            classifier.fit(X_train, Y_train)

            eval_test = {
                'f1': f1_score(Y_test, classifier.predict(X_test)),
                'confmat': confusion_matrix(Y_test, classifier.predict(X_test)),
                'dataset': dataset_name,
                'random_state': random_state
            }
            result_path = '/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/svm/'
            pd.DataFrame([eval_test]).to_csv(
                f'{result_path}best_model_{datetime.now().strftime("%Y-%m-%d-%-H:%M:%S")}.csv')
