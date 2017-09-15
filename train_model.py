import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras import backend as K

# from https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    total_memory = 4*(batch_size/1024**3)*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory, 3)
    return gbytes

# Constants
x_train_file_name = "train_pairs_x.npy"
y_train_file_name = "train_pairs_y.npy"

x_test_file_name = "test_pairs_x.npy"
y_test_file_name = "test_pairs_y.npy"

batch_size = 200
num_classes = 2
num_epoch = 100

img_rows = 64
img_cols = 64

# Load images
x_train = np.load(x_train_file_name)
y_train = np.load(y_train_file_name)

x_test = np.load(x_test_file_name)
y_test = np.load(y_test_file_name)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Set up output data in categorical matrices

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test, num_classes)

# Build the network model

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (1, 1)))
model.add(Activation("relu"))

model.add(Conv2D(512, (1, 1)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(num_classes))

model.add(Activation("softmax"))

preds = model.predict(np.ones((1, 64, 64, 2)))
print(preds.shape)

# Compile the model and put data between 0 and 1

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print("Memory usage:", get_model_memory_usage(batch_size, model), "GB")
print("Press enter if that's okay. If not, type NO and then press enter.")
if input() == "NO":
    exit()

print(x_train.shape)
print(y_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)

# Train the model

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(x_test, y_test),
              shuffle=True)

model.save("finished_model.hdf5")