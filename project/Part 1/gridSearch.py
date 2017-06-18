import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

K.set_image_dim_ordering('th')

X_train = np.load("data/X_train_small.npy")
y_train = load_labels("data/y_train_small.csv")
X_test = np.load("data/X_test.npy")

X_train = X_train[:100]
y_train = y_train[:100]

batch_size = 32
num_epochs = 3
kernel_size = 3
pool_size = 2
conv_depth_1 = 4
conv_depth_2 = 8
drop_prob_1 = 0.25
drop_prob_2 = 0.5

X_test_old_shape = X_test.shape
X_train = X_train.reshape(-1,3,32,32)
X_test = X_test.reshape(-1,3,32,32)

num_train, depth, height, width = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_train)

Y_train = np_utils.to_categorical(y_train, num_classes)

def create_model(neurons = 1, activation = 'relu',optimizer='adam'):
    inp = Input(shape=(depth, height, width))
    conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation=activation)(inp)
    conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation=activation)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation=activation)(drop_1)
    conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation=activation)(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    flat = Flatten()(drop_2)
    hidden = Dense(neurons, activation=activation)(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(outputs = out, inputs = inp)
    model = Sequential(layers=model.layers)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = create_model()

model = KerasClassifier(build_fn=create_model, verbose=0)
neurons = [8, 512]
activation = ['relu', 'sigmoid']
optimizer = ['adam', 'adagrad', 'sgd']
param_grid = dict(neurons=neurons, activation=activation,optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))