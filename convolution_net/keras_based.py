import os

import numpy as np
import tensorflow.keras
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential

ex = Experiment("atc: keras vgg")
ex.captured_out_filter = apply_backspaces_and_linefeeds

path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
ex.observers.append(MongoObserver(url=path))


class LogMetrics(Callback):
    def on_epoch_end(self, _, logs={}):
        train_metrics(logs=logs)
        validation_metrcis(logs=logs)

    def on_batch_end(self, _, logs={}):
        return


metrics = [
    tensorflow.keras.metrics.TruePositives(name='tp'),
    tensorflow.keras.metrics.FalsePositives(name='fp'),
    tensorflow.keras.metrics.TrueNegatives(name='tn'),
    tensorflow.keras.metrics.FalseNegatives(name='fn'),
    tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
    tensorflow.keras.metrics.Precision(name='precision'),
    tensorflow.keras.metrics.Recall(name='recall'),
    tensorflow.keras.metrics.AUC(name='auc'),
]


@ex.config
def cfg():
    base_batch_size = 128
    base_learning_rate = 0.1
    scale_batch_rate = 2
    epochs = 800
    early_stop_patience = 50


@ex.capture
def validation_metrcis(_run, logs):
    _run.log_scalar('val_loss', float(logs.get('val_loss')))
    _run.log_scalar('val_acc', float(logs.get('val_accuracy')))
    _run.log_scalar('val_auc', float(logs.get('val_auc')))
    _run.result = float(logs.get('val_auc'))


@ex.capture
def train_metrics(_run, logs):
    _run.log_scalar('train_loss', float(logs.get('loss')))
    _run.log_scalar('train_accuracy', float(logs.get('accuracy')))
    _run.log_scalar('train_auc', float(logs.get('auc')))


@ex.automain
def main(base_batch_size, base_learning_rate, scale_batch_rate, epochs, early_stop_patience, _run):
    files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train = np.load(train_path, allow_pickle=True)
    validation = np.load(validation_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    x_train, y_train = train['audio'], train['label']
    x_validation, y_validation = validation['audio'], validation['label']
    x_test, y_test = test['audio'], test['label']

    x_train = np.expand_dims(x_train, axis=-1)
    y_train = y_train
    x_validation = np.expand_dims(x_validation, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 63, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(1200, activation='relu'),
        Dropout(rate=0.5),
        Dense(640, activation='relu'),
        Dropout(rate=0.5),
        Dense(300, activation='relu'),
        Dropout(rate=0.5),
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    batch_size = base_batch_size * scale_batch_rate
    lr = base_learning_rate * scale_batch_rate
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200, verbose=1, mode='auto',
                                    min_delta=0.0001, cooldown=0, min_lr=0)
    sgd = optimizers.SGD(lr=lr)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patience,
                               verbose=1, mode='auto', restore_best_weights=True)

    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=metrics
    )

    model.fit(x=x_train, y=y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_validation, y_validation),
              callbacks=[lr_schedule, early_stop, LogMetrics()])
    model.save('vgg_unbalanced.h5')

    validation_prediction = model.predict_proba(x_validation)
    print(roc_auc_score(y_validation > .25, validation_prediction))
    print(confusion_matrix(y_validation > .25, validation_prediction > .35))
    test_prediction = model.predict_proba(x_test)
    print(roc_auc_score(y_test > .25, test_prediction))
    print(confusion_matrix(y_test > .25, test_prediction > .35))
    return f'auc={roc_auc_score(y_test > .25, test_prediction):.3}'

    # https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ -- for context inclusion
    # https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/ -- optimize training
