# coding: utf-8

#Import a bunch of stuff
import numpy as np
from keras import layers, callbacks
from keras.optimizers import SGD, Adam
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import dask.array as da
import scipy.misc
from matplotlib.pyplot import imshow
#This only works on jupyter notebook:
#get_ipython().magic('matplotlib inline')

#Clear Memory
import gc
gc.collect()

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#==========================================================================
#PART 0 - Set Some Stuff
#==========================================================================
img_width = 224
img_height = 224
classes = 196
batch_size = 16
epochs = 10 #Be sure to set this based on how long you want to run for!
patience = 50 #For Callbacks
verbose = 1
num_train_samples = 6549
num_valid_samples = 1695 #Cross validation: num_train_samples + num_valid_samples = # of train images
mode = 'sgd' #adam or sgd

#==========================================================================
#PART 1 - Initialize the Model
#==========================================================================

#Import the structure for an identity block, a convolution block, and a full residual CNN (based on both):
from identity_block import *
from conv_block import *
from res_net import *

#Build the model:
model = ResNet(input_shape = (img_width, img_height, 3), classes = classes)
if (mode == 'sgd'): optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True) #This one seems to work better
if (mode == 'adam'): optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
try:
	model.load_weights("weights.best.hdf5")
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
except:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#==========================================================================
#PART 2 - Import Data and Set-up Training Parameters (Callbacks, etc.)
#==========================================================================

# Load Dataset (note that test data is loaded separately lower down - this helps with memory)
X_train, Y_train_orig, classes = load_train_dataset()

# It's not necessary to normalize X_train because of the Scale step in the neural network (after BatchNorm)

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 196).T

# If cross validation is desired, then take some of the training data for this purpose
if (num_valid_samples > 0):
	X_valid = X_train[num_train_samples::,:,:,:]
	Y_valid = Y_train[num_train_samples::]
	X_train = X_train[0:num_train_samples,:,:,:]
	Y_train = Y_train[0:num_train_samples]
	

print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_valid shape: " + str(X_valid.shape))
print ("Y_valid shape: " + str(Y_valid.shape))

# Data Augmentation and Cross Validation
train_data_gen = image.ImageDataGenerator(rotation_range=20., width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
train_data_gen.fit(X_train)
train_generator = train_data_gen.flow(X_train, Y_train, batch_size=batch_size)
valid_data_gen = image.ImageDataGenerator()
valid_generator = valid_data_gen.flow(X_valid, Y_valid, batch_size=batch_size)

# Callbacks
checkpoint = callbacks.ModelCheckpoint("weights.best.hdf5", monitor='val_acc', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv_logger = callbacks.CSVLogger('training.log', append=True)
early_stop = callbacks.EarlyStopping('val_acc', patience=patience)
reduce_lr = callbacks.ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience/4), verbose=1)
callbacks=[csv_logger,checkpoint,early_stop,reduce_lr]

#==========================================================================
#PART 3 - Train the Model
#==========================================================================
model.fit_generator(train_generator, steps_per_epoch=num_train_samples/batch_size, validation_data=valid_generator, validation_steps=num_valid_samples/batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose)

#==========================================================================
#PART 4 - Test the Model
#==========================================================================

#Tidy up memory:
X_train_orig = Y_train_orig = X_train = Y_train = None

#Imoprt stuff
X_test, Y_test_orig, classes = load_test_dataset()
Y_test = convert_to_one_hot(Y_test_orig, 196).T

print ("number of test examples = " + str(X_test.shape[0]))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
