import tensorflow as tf
from librosa.feature import mfcc
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

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
    G = np.array([g for g in tqdm(register.file_id)])[:, None]
    Y = np.array([flatpool(np.load(p)) for p in tqdm(register.target_path.values)])[:, None]
    return X, G, Y


model = Sequential([
    Dense(512, activation='relu', input_shape=(660,)),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,
                                                  patience=100, mode='max',
                                                  restore_best_weights=True)

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=METRICS)

from convolution_net.load import build_register, train_dev_test

register = build_register(datasets['trainspeed_2sec'])
random_state = 0  # np.random.randint(0, 1000))
registers = train_dev_test(register, random_state=random_state)

X_train, G_train, Y_train = load(registers['train'])
X_dev, G_dev, Y_dev = load(registers['dev'])
X_test, G_test, Y_test = load(registers['test'])

# noramlizer
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


# oversampling
def oversample(X, Y):
    pos_idx = (Y != 0).squeeze()
    X_pos = X[pos_idx]
    X_neg = X[~pos_idx]
    Y_pos = Y[pos_idx]
    Y_neg = Y[~pos_idx]

    ids = len(X_pos)
    choices = np.random.choice(ids, len(X_neg))
    X_pos = X_pos[choices]
    Y_pos = Y_pos[choices]

    X = np.concatenate([X_pos, X_neg], axis=0)
    Y = np.concatenate([Y_pos, Y_neg], axis=0)

    order = np.arange(len(X))
    np.random.shuffle(order)
    X = X[order]
    Y = Y[order]
    return X, Y


X_train, Y_train = oversample(X_train, Y_train)

model.fit(X_train, Y_train, callbacks=[early_stopping], validation_data=(X_dev, Y_dev), epochs=2000)
print(confusion_matrix(Y_test, model.predict(X_test) > 0.5))
print(f1_score(Y_test, model.predict(X_test) > 0.5))
