import os

import numpy as np
import tensorflow.keras as keras
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential

ex = Experiment("atc: keras vgg no dropout")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


def make_model(lr, output_bias):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
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
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(640, activation='relu'),
        Dropout(rate=0.5),
        Dense(300, activation='relu'),
        Dropout(rate=0.5),
        Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=metrics)

    return model


@ex.config
def cfg():
    base_batch_size = 128
    base_learning_rate = 0.1
    scale_batch_rate = 2
    epochs = 800
    early_stop_patience = 150


@ex.capture
def validation_metrics(_run, logs):
    _run.log_scalar('val_loss', float(logs.get('val_loss')))
    _run.log_scalar('val_acc', float(logs.get('val_accuracy')))
    _run.log_scalar('val_auc', float(logs.get('val_auc')))
    _run.result = f"auc={logs.get('val_auc'):.3}"


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
    x_validation = np.expand_dims(x_validation, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = y_train > 0.25
    y_validation = y_validation > 0.25
    y_test = y_test > 0.25

    n_true, n_false = np.bincount(y_train)
    n_total = len(y_train)
    weight_for_0 = (1 / n_false) * (n_total) / 2.0
    weight_for_1 = (1 / n_true) * (n_total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    batch_size = base_batch_size * scale_batch_rate

    lr = base_learning_rate * scale_batch_rate
    lr_schedule = ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=100, verbose=1, mode='max',
                                    min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = EarlyStopping(monitor='val_auc', min_delta=0, patience=early_stop_patience,
                               verbose=1, mode='max', restore_best_weights=True)


    model = make_model(lr=lr, output_bias=np.log(n_true / n_false))
    model.summary()

    model.fit(x=x_train, y=y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_validation, y_validation),
              callbacks=[lr_schedule, early_stop],
              class_weight=class_weight)
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
