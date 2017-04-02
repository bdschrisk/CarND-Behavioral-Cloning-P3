# Adapted from github.com/titu1994/DenseNet

from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import keras.backend as K

# import custom keras layers
from extensions.layers.core_extended import Resize

### Model Functions ###

# Normalizes the input using min-max normalization.
def Normalizer(x):
    return (x / 255) - 0.5

def Normalizer_shape(input_shape):
    return input_shape

def Convolution(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional bottleneck block and dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def TransitionBlock(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x

def DenseBlock(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = Convolution(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter

# Returns an input tensor for the rest of the network
# - input_shape: image shape shape
# - crop_width: 2D tuple of cropping from left and right edges
# - crop_height: 2D tuple of cropping from top and bottom edges
# - resize_factor: scaling factor to apply to the input after cropping
# - sigma: Scalar value of the gaussian noise function.
def InputLayer(input_shape, crop_width = (0,0), crop_height = (50,20), resize_factor = 0.5, sigma = 0.1):
    # Initialise input network
    input = Input(shape=input_shape)
    # Reshape for Cropping layer
    #model = Reshape((input_shape[2], input_shape[1], input_shape[0]))(input)
    # Apply cropping
    model = Cropping2D(cropping=(crop_height, crop_width))(input)
    # Sample wise min-max normalization layer
    model = Lambda(Normalizer, output_shape=Normalizer_shape)(model)
    # Calculate new dimensions after cropping
    new_height = (input_shape[0] - (crop_height[0] + crop_height[1])) * resize_factor
    new_width = (input_shape[1] - (crop_width[0] + crop_width[1])) * resize_factor
    # add Resize layer (CUSTOM)
    model = Resize((int(new_height), int(new_width)), axis=(1, 2), interpolation='nearest_neighbor')(model)
    
    if (sigma > 0):
        # Apply noise
        model = GaussianNoise(sigma)(model)
    
    return (input, model)

def DenseNet(img_dim, resize_factor, nb_classes, output_activation = 'softmax', depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
             bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4, noise = 0, verbose=True):
    ''' Build the DenseNet model

    Args:
        img_dim: input tensor shape
        resize_factor: image resizing / scaling factor
        nb_classes: number of class labels
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        verbose: print the model type

    Returns: 3D tuple of Input, Output and Classification (final output) layers

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    if bottleneck:
        nb_layers = int(nb_layers // 2)

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction
    
    (input, output) = InputLayer(img_dim, resize_factor = resize_factor, sigma = noise)

    # Initial convolution
    output = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(output)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        output, nb_filter = DenseBlock(output, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        output = TransitionBlock(output, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    output, nb_filter = DenseBlock(output, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    output = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(output)
    output = Activation('relu')(output)
    output = GlobalAveragePooling2D()(output)
    output = Dense(nb_classes, activation=output_activation, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(output)

    if verbose:
        if bottleneck and not reduction:
            print("Bottleneck DenseNet-B-%d-%d created." % (depth, growth_rate))
        elif not bottleneck and reduction > 0.0:
            print("DenseNet-C-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        elif bottleneck and reduction > 0.0:
            print("Bottleneck DenseNet-BC-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        else:
            print("DenseNet-%d-%d created." % (depth, growth_rate))
    
    # return input, intermediate and output layers
    model = Model(input = input, output = output)
    return model