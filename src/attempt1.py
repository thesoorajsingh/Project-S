#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
from string import ascii_lowercase

print("Using GPU -> " + str(tf.test.is_gpu_available()))

# %%
dataset_path = "../data/ASL-data"
train_folder = list(os.walk(f"{dataset_path}/train"))[0][2]
val_folder = list(os.walk(f"{dataset_path}/valid"))[0][2]
test_folder = list(os.walk(f"{dataset_path}/test"))[0][2]


def read_images_to_dataset(paths, suffix):
    train_images = []
    train_labels = []
    for path in paths:
        if path[0].lower() in ascii_lowercase:
            img = tf.io.decode_jpeg(tf.io.read_file(f"{dataset_path}/{suffix}/{path}"))
            # reshape images to be of same resolution
            train_images.append(tf.image.resize(img, (372, 372)).numpy())
            train_labels.append(ascii_lowercase.index(path[0].lower()))

    train_images = np.stack(train_images)
    train_labels = np.array(train_labels)
    # train_labels = tf.one_hot(train_labels, len(ascii_lowercase)).numpy()
    return train_images / 255.0, train_labels


x_train, y_train = read_images_to_dataset(train_folder, "train")
x_val, y_val = read_images_to_dataset(val_folder, "valid")
x_test, y_test = read_images_to_dataset(test_folder, "test")

print(f"Train data shape -> {x_train.shape}")
print(f"Train data shape -> {x_val.shape}")
print(f"Train data shape -> {x_test.shape}")

#%%
# functions to see training data
take = 3
for t in range(take):
    a = np.random.randint(0, x_train.shape[0], 1).item()
    print(f"Letter of the alphabet -> {ascii_lowercase[y_train[a]].upper()}")
    plt.imshow(x_train[a])
    plt.show()
    print()

# %%
def create_model():
    def double_conv_layer(y):
        x = Conv2D(64, 3, padding="same")(y)
        x = Add()([x, y])
        x = Conv2D(64, 3, padding="same", activation="gelu")(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        return x

    inp = Input(shape=(372, 372, 3))

    x = BatchNormalization()(inp)
    x = Conv2D(64, 3, activation="gelu", padding="same")(x)
    x = Conv2D(64, 3, activation="gelu", padding="same")(x)
    x = MaxPool2D()(x)
    x = double_conv_layer(x)
    x = double_conv_layer(x)
    x = double_conv_layer(x)
    x = double_conv_layer(x)
    x = double_conv_layer(x)
    x = double_conv_layer(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(len(ascii_lowercase), activation="softmax")(x)

    return Model(inputs=inp, outputs=x)


model = create_model()
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
# %%
model.fit(x_train, y_train, batch_size=8, epochs=100, validation_data=(x_val, y_val))
# %%
