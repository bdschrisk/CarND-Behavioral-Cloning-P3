import os
import numpy as np
import pandas as pd
import csv

# import Keras modules
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from callbacks import EarlyStoppingByDivergence

import modelutils as mut

import modelconfig as config
import generator as g

import kanet as kn
import densenet as dn

### Model Script ###

print("")
### Data ###
train_file = os.path.join(config.feature_training_dir, "training_data.csv")
test_file = os.path.join(config.feature_training_dir, "testing_data.csv")

# initialise datasets
X_train, y_train, X_test, y_test = None, None, None, None

# load previously saved samples if available
if (os.path.exists(train_file) and os.path.exists(test_file)):
    print("Initialising datasets...")
    # load datasets
    train_samples = np.array(pd.read_csv(train_file, sep=','))
    test_samples = np.array(pd.read_csv(test_file, sep=','))
    # extract into training / testing matrices
    X_train = train_samples[:, 0]
    y_train = train_samples[:, 1:].astype(np.float)
    X_test = test_samples[:, 0]
    y_test = test_samples[:, 1:].astype(np.float)
    print(" - Done.\n")
else:
    # load samples for feeding to generators
    print("Constructing datasets...")
    samples = []
    with open(config.feature_training_file) as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        if (config.feature_training_skip_header):
            next(reader) 
        # read samples
        for sample in reader:
            # is relative path
            if (config.feature_training_image_relative): 
                name = os.path.join(config.feature_training_dir, sample[config.feature_training_image_col].split('/')[-1])
            else: name = sample[config.feature_training_image_col]
            
            if os.path.exists(name) and len(sample[config.feature_training_steering_col]) > 0:
                samples.append([name, float(sample[config.feature_training_steering_col]), float(sample[config.feature_training_throttle_col])])

    ### Input sampling ###
    samples = np.array(samples)
    X = samples[:, 0]
    y = samples[:, 1:].astype(np.float)
    print("Shape X = {}, Shape y = {}".format(X.shape, y.shape))
    # train / test split
    X_train, y_train, X_test, y_test = mut.sequential_split(X, y, config.batch_size, config.train_split)
    print("Shape X_train = {}, Shape y_train = {}".format(np.array(X_train).shape, np.array(y_train).shape))
    print("Shape X_test = {}, Shape y_test = {}".format(np.array(X_test).shape, np.array(y_test).shape))
    # compress samples into shape for saving
    train_samples, test_samples = np.column_stack((X_train, y_train)), np.column_stack((X_test, y_test))
    # save datasets for resuming training
    with open(train_file, mode = "wb") as file:
        np.savetxt(file, train_samples, delimiter=",", fmt = "%s")
    with open(test_file, mode = "wb") as file:
        np.savetxt(file, test_samples, delimiter=",", fmt = "%s")
    print(" - Done.\n")

print("{} Samples loaded.".format(len(y_train)+len(y_test)))
print("Training size = {}, Validation size = {}".format(len(X_train), len(X_test))) 


### Model ###

print("\nInitialising model...")
# Create the model
model = kn.KaNet(config.img_dim, config.resize_factor, config.output_size, config.output_activation, config.dropout, config.weight_decay, 0)
#model = dn.DenseNet(config.img_dim, config.resize_factor, config.output_size, output_activation = config.output_activation, 
#                    depth = config.depth, growth_rate = config.growth_rate, bottleneck = config.bottleneck, reduction = config.reduction)
                    
# compile the model using the specified optimizer and loss / metrics
model.compile(config.optimiser, config.loss, metrics=config.metrics, loss_weights=None, sample_weight_mode=None)

epoch = 0
# Load existing weights (if found)
checkpoints = mut.get_checkpoints(config.checkpoint_pattern)
if (len(checkpoints) > 0):
    print("\nLoading previous checkpoint...")
    checkpoint = checkpoints[0, :]
    epoch = int(checkpoint[1])
    model.load_weights(checkpoint[0])
    print(" - Weights restored. (epoch = {}, loss = {})".format(checkpoint[1], checkpoint[2]))


### Train ###

print("\nInitializing generators...")

# compile and train the model using the generator function
train_generator = g.generator(X_train, y_train, batch_size = config.batch_size, noise = config.noise, sequential = False)
validation_generator = g.generator(X_test, y_test, batch_size = config.batch_size, noise = 0, sequential = False)
print(" - Done.\n")

# setup callbacks, save every checkpoint, check for overfitting and reduce learning rate as required
callbacks = [
    CSVLogger("./model/training-loss.csv", separator=',', append = True),
    #ReduceLROnPlateau(monitor='val_loss', factor = 0.1, cooldown = 0, patience = 2, min_lr = 10e-6), # if using non-adaptive optimizer
    EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience = 7),
    EarlyStoppingByDivergence(patience = 8),
    ModelCheckpoint(filepath = config.checkpoint_pattern.replace("*","{epoch:02d}-{val_loss:.2f}"), monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=0),
]

print("\n## Model ##")
print(model.summary())

nb_train_samples = (config.epoch_train_samples if config.epoch_train_samples > 0 else len(y_train))
nb_val_samples = (config.epoch_test_samples if config.epoch_test_samples > 0 else len(y_test))

print("\nTraining...")
# Train the model using the supplied generator
history = model.fit_generator(train_generator, samples_per_epoch = config.epoch_train_samples, validation_data = validation_generator, nb_val_samples = config.epoch_test_samples,
                              nb_epoch = config.max_epochs, callbacks = callbacks, initial_epoch = epoch)

