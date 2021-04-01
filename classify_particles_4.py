dir_suffix="ve_va_ym"
my_path="YOUR\\PATH\\HERE"

"""
#######PREPROCESSING#######
"""
import numpy as np
from PIL import Image
import time

def binarize(grayscale, thresh):
    return np.where(grayscale > thresh, 1, 0)

def get_data_image(filename, bin_thresh=0.85):
    img = Image.open(filename).resize((720, 720)).convert("LA")
    image = np.asarray(img)/256
    binarized = binarize(image, bin_thresh)
    ret_img = binarized[:, :, 0:1]
    return ret_img

"""
#######BUILD MODEL#######
"""
from keras import layers
from keras import models
from keras import optimizers

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 720, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc']

    )
    return model

def other_build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(720, 720, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc']

    )
    return model

"""
#######TRAINING#######
"""

import tensorflow as tf
from keras.utils import Sequence

"""Data generator for Keras"""
class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(720,720), n_channels=1,
                 n_classes=None, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    """Returns the # of batches per epoch"""
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    """
    Generates a batch of data at a given index
    """
    def __getitem__(self, index):
        # Get indices of batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get list of IDs to generate data from
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    """Set indices after epoch to randomize data"""
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    """
    Generate data for a single batch
    """
    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            X[i,] = self.get_data_image(ID)

            # Store class
            y[i] = self.labels[ID]

        return X, y
    
    def binarize(self, grayscale, thresh):
        return np.where(grayscale > thresh, 1, 0)

    def get_data_image(self, filename, bin_thresh=0.85):
        img = Image.open(filename).resize(self.dim).convert("LA")
        image = np.asarray(img)/256
        binarized = self.binarize(image, bin_thresh)
        ret_img = binarized[:, :, 0:1]
        return ret_img

"""Set up training and validation data"""
train_x =  [my_path+"\\detector_alpha_%s\\plots\\%d.png"     % (dir_suffix, i) for i in range(4000)]
train_x += [my_path+"\\detector_electrons_%s\\plots\\%d.png" % (dir_suffix, i) for i in range(4000)]
valid_x =  [my_path+"\\detector_alpha_%s\\plots\\%d.png"     % (dir_suffix, i) for i in range(4000,5000)]
valid_x += [my_path+"\\detector_electrons_%s\\plots\\%d.png" % (dir_suffix, i) for i in range(4000,5000)]
# There's probably a better way to do this, but oh well for now.
train_label_map = {filename : (1 if "detector_alpha_" in filename else 0) for filename in train_x}
valid_label_map = {filename : (1 if "detector_alpha_" in filename else 0) for filename in valid_x}

params = {'dim': (720,720),
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Set up generators
training_generator =   DataGenerator(train_x, train_label_map, **params)
validation_generator = DataGenerator(valid_x, valid_label_map, **params)

# Fit!
model = other_build_model()
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    verbose=True,
                    epochs=40)

hist_dict = history.history

def save_history_data(hist_dict):
    for key in hist_dict.keys():
        with open("%s\\%s_%s"%(my_path, key, dir_suffix), "w+") as f:
            for value in hist_dict[key]:
                f.write(str(value)+"\n")

save_history_data(hist_dict)


model.save("%s\\model_%s.h5"%(my_path, dir_suffix))

import matplotlib.pyplot as plt

plt.xlabel("Epochs")
plt.ylabel("Loss")
epochs=range(1,len(hist_dict["loss"])+1)
plt.plot(epochs, hist_dict["loss"], "bo", label="Training loss")
plt.plot(epochs, hist_dict["val_loss"], "b", label="Validation loss")
plt.title("Loss values")
plt.legend()
plt.savefig(my_path+"\\%s_loss.png" % (dir_suffix))

plt.clf()

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
epochs=range(1,len(hist_dict["acc"])+1)
plt.plot(epochs, hist_dict["acc"], "bo", label="Training accuracy")
plt.plot(epochs, hist_dict["val_acc"], "b", label="Validation accuracy")
plt.title("Accuracy")
plt.legend()
plt.savefig("%s\\%s_accuracy.png" % (my_path, dir_suffix))
