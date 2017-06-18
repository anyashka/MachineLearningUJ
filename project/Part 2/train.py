import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, UpSampling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

def save_labels(arr, filename):
    pd_array = pd.DataFrame(arr)
    pd_array.index.names = ["Id"]
    pd_array.columns = ["Prediction"]
    pd_array.to_csv(filename)

def load_labels(filename):
    return pd.read_csv(filename, index_col=0).values.ravel()

def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

X_train = np.load("data/X_train.npy")
X_train_small = np.load("data/X_train_small.npy")
y_train_small = load_labels("data/y_train_small.csv")
X_test = np.load("data/X_test.npy")

X_train = X_train.reshape(-1,3,32,32)
X_train_small = X_train_small.reshape(-1,3,32,32)
X_test = X_test.reshape(-1,3,32,32)

X_train = X_train.astype('float32')
X_train_small = X_train_small.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_train_small /= np.max(X_train_small)
X_test /= np.max(X_train)

num_train, depth, height, width = X_train_small.shape

num_test = X_test.shape[0]
num_classes = np.unique(y_train_small).shape[0]

Y_train_small = np_utils.to_categorical(y_train_small, num_classes)

batch_size = 32

input_img = Input(shape=(depth, height, width))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,
                epochs=5,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_train_small, X_train_small))

num_epochs = 2

x = autoencoder.output
x = Flatten()(input_img)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)

model = Model(outputs = x, inputs = input_img)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_small, Y_train_small,
          batch_size=batch_size, epochs = 200,
          verbose=1, validation_split=0.1)
train_acc_1 = model.evaluate(X_train_small, Y_train_small, verbose=1)
print("Train acc:", train_acc_1)
y_proba = model.predict(X_test)

save_labels(probas_to_classes(y_proba), "data/y_pred_small_unsuper.csv")

#second model without preprocessing

x = Flatten()(input_img)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)

model_2 = Model(outputs = x, inputs = input_img)

model_2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_2.fit(X_train_small, Y_train_small,
          batch_size=batch_size, epochs = 200,
          verbose=1, validation_split=0.1)
train_acc_2 = model_2.evaluate(X_train_small, Y_train_small, verbose=1)
print("Train acc 2:", model.evaluate(X_train_small, Y_train_small, verbose=1))
y_proba_2 = model.predict(X_test)

save_labels(probas_to_classes(y_proba_2), "data/y_pred_small_2.csv")

print("Train acc check1:", train_acc_1[1])
print("Train acc check2:", train_acc_2[1])
