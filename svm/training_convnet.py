import numpy as np
import tensorflow as tf
from librosa.feature import melspectrogram
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from convolution_net.load import build_register, train_dev_test


class Normalizer:
    def __init__(self):
        self.xm = float()
        self.xv = float()

    def fit(self, X):
        self.xm = X.mean()
        self.xv = X.var()

    def transform(self, X):
        X = X - self.xm
        X = X / self.xv
        return X

    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X


# oversampling
def oversample(X, Y):
    pos_idx = (Y != 0).squeeze()
    X_pos, Y_pos = X[pos_idx], Y[pos_idx]
    X_neg, Y_neg = X[~pos_idx], Y[~pos_idx]

    choices = np.random.choice(len(X_pos), len(X_neg))
    X_pos, Y_pos = X_pos[choices], Y_pos[choices]

    X = np.concatenate([X_pos, X_neg], axis=0)
    Y = np.concatenate([Y_pos, Y_neg], axis=0)

    order = np.arange(len(X))
    np.random.shuffle(order)
    X, Y = X[order], Y[order]
    return X, Y


def load(register):
    mels = lambda x: np.log(7 * melspectrogram(x, sr=8192, n_fft=512, hop_length=128, n_mels=40))
    flatpool = lambda x: np.mean(x) > 0.05

    X = np.array([mels(np.load(p)) for p in tqdm(register.audio_path.values)])[:, :, :, None]
    Y = np.array([flatpool(np.load(p)) for p in tqdm(register.target_path.values)])[:, None]
    return X, Y


datasets = {
    'trainspeed_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_2sec',
    'trainpseed_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_5sec',
    'subsample_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_2sec',
    'subsample_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_5sec',
    'beatfrequency_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_2sec',
    'beatfrequency_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_5sec',
}

from tensorflow.keras.regularizers import l2


def make_model(n_filter, l2_ratio, dropout_ratio, learning_rate):
    model = Sequential([
        Conv2D(n_filter, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same',
               input_shape=(40, 129, 1)),
        # BatchNormalization(axis=-1),
        # Conv2D(n_filter, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same'),
        # BatchNormalization(axis=-1),
        MaxPool2D((2, 2)),
        Dropout(dropout_ratio / 2),
        Conv2D(n_filter * 2, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same'),
        # BatchNormalization(axis=-1),
        # Conv2D(n_filter*2, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same'),
        # BatchNormalization(axis=-1),
        MaxPool2D((2, 2)),
        Dropout(dropout_ratio / 2),
        Conv2D(n_filter * 4, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same'),
        # BatchNormalization(axis=-1),
        # Conv2D(n_filter*4, (3, 3), activation='relu', kernel_regularizer=l2(l2_ratio), padding='same'),
        # BatchNormalization(axis=-1),
        MaxPool2D((2, 2)),
        Dropout(dropout_ratio / 2),
        Flatten(),

        Dense(512, activation='relu', kernel_regularizer=l2(l2_ratio)),
        # BatchNormalization(axis=-1),
        Dropout(dropout_ratio),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_ratio)),
        # BatchNormalization(axis=-1),
        Dropout(dropout_ratio),
        Dense(1, activation='sigmoid'),
    ])

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=METRICS)
    return model


model = make_model(n_filter=64, l2_ratio=0.0001, dropout_ratio=0.2, learning_rate=0.01)

register = build_register(datasets['trainspeed_2sec'])
random_state = 0  # np.random.randint(0, 1000))
registers = train_dev_test(register, random_state=random_state)

X_train, Y_train = load(registers['train'])
X_dev, Y_dev = load(registers['dev'])
X_test, Y_test = load(registers['test'])

normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_dev = normalizer.transform(X_dev)
X_test = normalizer.transform(X_test)
X_train, Y_train = oversample(X_train, Y_train)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,
                                                  patience=10, mode='max',
                                                  restore_best_weights=True)
model.fit(X_train, Y_train, epochs=2000,
          callbacks=[early_stopping], validation_data=(X_dev, Y_dev),
          class_weight=[2.0, 1.0])
print(confusion_matrix(Y_test, model.predict(X_test) > 0.5))
print(f1_score(Y_test, model.predict(X_test) > 0.5))
