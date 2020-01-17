import os

import numpy as np
from sacred import Experiment
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

from conv_models import make_model, metrics, RocCallback
from data_preprocessing import upsample_minority

ex = Experiment("atc: Small_VGG")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))

@ex.config
def cfg():
    batch_size = 256
    learning_rate = 0.2
    epochs = 800
    early_stop_patience = 200
    model_name = 'CustomVGG'


class LogMetrics(Callback):
    def on_epoch_end(self, _, logs={}):
        train_metrics(logs=logs)
        validation_metrcis(logs=logs)

    def on_batch_end(self, _, logs={}):
        return

@ex.capture
def validation_metrcis(_run, logs):
    _run.log_scalar('val_loss', float(logs.get('val_loss')))
    _run.log_scalar('val_acc', float(logs.get('val_accuracy')))
    _run.log_scalar('val_auc_keras', float(logs.get('val_auc')))


@ex.capture
def train_metrics(_run, logs):
    _run.log_scalar('train_loss', float(logs.get('loss')))
    _run.log_scalar('train_accuracy', float(logs.get('accuracy')))
    _run.log_scalar('train_auc_keras', float(logs.get('auc')))

@ex.automain
def main(model_name, batch_size, learning_rate,
         epochs, early_stop_patience, _run):
    files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train = np.load(train_path, allow_pickle=True)
    validation = np.load(validation_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    x_train, y_train = train['audio'], train['label']
    x_validation, y_validation = validation['audio'], validation['label']
    x_test, y_test = test['audio'], test['label']

    # reshape for convolution
    x_train = np.expand_dims(x_train, axis=-1)
    x_validation = np.expand_dims(x_validation, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train, y_train = upsample_minority(x_train, y_train)

    model = make_model(model_name)
    model.summary()

    lr_schedule = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=20, verbose=1, mode='auto',
                                    min_delta=0.0001, cooldown=10, min_lr=0)
    sgd = optimizers.SGD(lr=learning_rate)
    roc_callback = RocCallback(training_data=(x_train, y_train), validation_data=(x_test, y_test), _run=_run)
    early_stop = EarlyStopping(monitor='auc', min_delta=0, patience=early_stop_patience,
                               verbose=1, mode='auto', restore_best_weights=True)

    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=metrics
    )

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_validation, y_validation),
              callbacks=[lr_schedule, early_stop, roc_callback, LogMetrics()])
    model.save('vgg_unbalanced_np.h5')

    validation_prediction = model.predict_proba(x_validation)
    print(roc_auc_score(y_validation > .25, validation_prediction))
    print(confusion_matrix(y_validation > .25, validation_prediction > .35))
    test_prediction = model.predict_proba(x_test)
    print(roc_auc_score(y_test > .25, test_prediction))
    print(confusion_matrix(y_test > .25, test_prediction > .35))
    return f'auc={roc_auc_score(y_test > .25, test_prediction):.3}'
