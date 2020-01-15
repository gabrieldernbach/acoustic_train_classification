import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten


def make_model(model_name, output_bias=None):
    constructor_dict = {
        'ResNet50': construct_ResNet50,
        'MobileNetV2': construct_MobileNetV2,
        'CustomVGG': constructor_customVgg,
        'tiny_VGG': tiny_Vgg
    }

    constructor = constructor_dict[model_name]
    model = constructor(output_bias)
    return model


def construct_ResNet50(output_bias):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    resnet50 = ResNet50(include_top=True, input_shape=(128, 63, 3), classes=10, weights=None)

    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                     input_shape=(128, 63, 1)))
    model.add(resnet50)
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))

    return model


def construct_MobileNetV2(output_bias):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    mobilenet = MobileNetV2(input_shape=(128, 63, 3), alpha=1.0, include_top=True, weights=None,
                            input_tensor=None, pooling=None, classes=10)

    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                     input_shape=(128, 63, 1)))
    model.add(mobilenet)
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))

    return model


def constructor_customVgg(output_bias):
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
    return model


def tiny_Vgg(output_bias):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
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
        Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    return model

## dummy for model finetuning
# base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
# x = keras.layers.GlobalAveragePooling2D()(base_model.output)
# output = keras.layers.Dense(n_classes, activation='softmax')(x)
# model = keras.models.Model(inputs=[base_model.input], outputs=[output])
