import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
from keras import optimizers
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K

from sklearn.metrics import roc_auc_score, confusion_matrix

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
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(640, activation='relu'),
    Dropout(rate=0.5),
    Dense(300, activation='relu'),
    Dropout(rate=0.5),
    Dense(1, activation='sigmoid')
])

for layer in model.layers:
    print(layer.output_shape)

base_batch_size = 256
base_lr = 0.1
multiplier = 2
batch_size = base_batch_size * multiplier
lr = base_lr * multiplier

sgd = optimizers.SGD(lr=lr)

es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=8,
                   verbose=0, mode='auto')


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


model.compile(
    loss='binary_crossentropy',
    optimizer=sgd,
    metrics=['accuracy', auc]
)

history = model.fit(x=x_train, y=y_train,
                    batch_size=batch_size,
                    epochs=200,
                    validation_data=(x_validation, y_validation),
                    callbacks=[es])

# import matplotlib.pyplot as plt
#
# plt.plot(history.history['acc'], label='training accuracy')
# plt.plot(history.history['val_acc'], label='testing accuracy')
# plt.title('Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()

validation_prediction = model.predict_proba(x_validation)
print(roc_auc_score(y_validation > .35, validation_prediction))
print(confusion_matrix(y_validation > .35, validation_prediction > .35))
test_prediction = model.predict_proba(x_test)
print(roc_auc_score(y_test > .35, test_prediction))
print(confusion_matrix(y_test > .35, test_prediction > .35))

# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
