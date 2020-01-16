import tensorflow
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, GlobalAvgPool2D, BatchNormalization


def make_model(model_name):
    constructor_dict = {
        'ResNet50': construct_ResNet50,
        'MobileNetV2': construct_MobileNetV2,
        'CustomVGG': constructor_customVgg,
        'tiny_VGG': tiny_Vgg
    }

    constructor = constructor_dict[model_name]
    model = constructor()
    return model


def construct_ResNet50():
    resnet50 = ResNet50(include_top=True, input_shape=(128, 63, 3), classes=10, weights=None)

    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                     input_shape=(128, 63, 1)))
    model.add(resnet50)
    model.add(Dense(1, activation='sigmoid'))

    return model


def construct_MobileNetV2():
    mobilenet = MobileNetV2(input_shape=(128, 63, 3), alpha=1.0, include_top=True, weights=None,
                            input_tensor=None, pooling=None, classes=10)

    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                     input_shape=(128, 63, 1)))
    model.add(mobilenet)
    model.add(Dense(1, activation='sigmoid'))

    return model


def constructor_customVgg():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(128, 63, 1)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        # Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),  # 63, 32
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        # Dropout(rate=0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),  # 32, 16
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        # Dropout(rate=0.25),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),  # 16, 8
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),  # 8, 4
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),  # 4, 2
        BatchNormalization(),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAvgPool2D(),
        # Dropout(rate=0.25),
        Flatten(),
        Dense(640, activation='relu'),
        BatchNormalization(),
        # Dropout(rate=0.5),
        Dense(300, activation='relu'),
        BatchNormalization(),
        # Dropout(rate=0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def tiny_Vgg():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 63, 1)),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(640, activation='relu'),
        Dense(300, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


## dummy for inhereting model
# base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
# x = keras.layers.GlobalAveragePooling2D()(base_model.output)
# output = keras.layers.Dense(n_classes, activation='softmax')(x)
# model = keras.models.Model(inputs=[base_model.input], outputs=[output])


# Model Metrics

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


class RocCallback(Callback):
    def __init__(self, training_data, validation_data, _run):
        super(RocCallback, self).__init__()
        self.x = training_data[0]
        self.y = training_data[1] > .25
        self.x_val = validation_data[0]
        self.y_val = validation_data[1] > .25
        self._run = _run

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train, 4)), str(round(roc_val, 4))),
              end=100 * ' ' + '\n')
        self._run.log_scalar('train_auc', float(roc_train))
        self._run.log_scalar('validation_auc', float(roc_val))
        self._run.result = float(roc_val)
        return
