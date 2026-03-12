#  Architectural design inspired by MobileNetV1
#  Taken from https://medium.com/data-science/building-mobilenet-from-scratch-using-tensorflow-ad009c5dd42c

import tensorflow as tf
import numpy as np  
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Rescaling, Dropout
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model

NUM_CLASSES = 10

#evita alocar toda a memória da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def mobilenetBlock(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

inputs = Input(shape = (128,128,3))
x = Rescaling(1./255)(inputs)
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = mobilenetBlock(x, filters = 64, strides = 1)
x = mobilenetBlock(x, filters = 128, strides = 2)
x = mobilenetBlock(x, filters = 128, strides = 1)
x = mobilenetBlock(x, filters = 256, strides = 2)
x = mobilenetBlock(x, filters = 256, strides = 1)
x = mobilenetBlock(x, filters = 512, strides = 2)

for _ in range (5):
     x = mobilenetBlock(x, filters = 512, strides = 1)
x = mobilenetBlock(x, filters = 1024, strides = 2)
x = mobilenetBlock(x, filters = 1024, strides = 1)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x) # Adicionando dropout para reduzir overfitting
output = Dense (units = NUM_CLASSES, activation = 'softmax')(x) #10 classes 
model = Model(inputs=inputs, outputs=output)
model.summary()

train_ds = tf.keras.utils.image_dataset_from_directory(
    "croppedImages_aug/train",
    image_size=(128, 128),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "croppedImages_aug/val",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

labels = np.array([y.numpy() for x, y in train_ds.unbatch()])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

print(class_weights)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    class_weight=class_weights
)


# train loss desce mas não tem melhora na val accuracy, provavelmente overfitting.
