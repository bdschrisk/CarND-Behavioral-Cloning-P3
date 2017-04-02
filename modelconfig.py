### Model Configuration File ###

### Pretraining config ###
feature_training_dir = "./data/TRAIN/"
feature_training_image_dir = feature_training_dir + "/IMG"
feature_training_file = feature_training_dir + "/driving_log.csv"
feature_training_skip_header = True
feature_training_throttle_col = 4
feature_training_steering_col = 3
feature_training_image_col = 0
feature_training_image_relative = True

### Model config ###
# models path
model_path = "./model/"
# model checkpoint path pattern, used for restoring models and saving checkpoints in training
checkpoint_pattern = model_path + "checkpoint-*.h5"

# returns the class ranges, using -1 to +1 interval
output_size = 2
img_dim = (160, 320, 3)
resize_factor = 0.5

### Feature Detection Network params ###
dropout = 0.2
layers = [256, 128]
depth = 22
growth_rate = 12
bottleneck = True
reduction = 0.5
output_activation = "tanh"

### Training parameters ###
train_split = 0.7
noise = 0.3
batch_size = 32
max_epochs = 60
epoch_train_samples = batch_size * 10 * 4 * 3
epoch_test_samples = batch_size * 10 * 4

# Optimiser params
optimiser = "Adam"
loss = "mse"
metrics = ["mae"]

learning_rate = 0.001
momentum = 0.9
weight_decay = 0.001
use_nesterov = True