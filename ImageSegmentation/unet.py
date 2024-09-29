import numpy as np
import tensorflow as tf

def encoder_conv_block(c_layer, n_filters, kernel_size = 3, dropout_prob = 0, max_pooling = True):
    """
    Function that implements a convolutional downsampling block.

    Arguments:
    c_layer -- Current tensor
    n_filters -- Number of filters for the convolutional layers
    kernel_size -- Size of applied filters
    dropout_prob -- Dropout probability 
    max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume

    Returns:
        next_layer, skip_connnection -- Next layer and skip connection outputs
    """

    n_layer = tf.keras.layers.Conv2D(n_filters,
                                     kernel_size,
                                     activation='relu',
                                     padding='same',
                                     kernel_initializer='he_normal')(c_layer)
    n_layer = tf.keras.layers.Conv2D(n_filters,
                                     kernel_size,
                                     activation='relu',
                                     padding='same',
                                     kernel_initializer='he_normal')(n_layer)


    # If dropout_prob > 0 add a dropout layer
    if dropout_prob > 0:
        n_layer = tf.keras.layers.Dropout(dropout_prob)(n_layer)

    skip_connection = n_layer
    
    # If max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        n_layer = tf.keras.layers.MaxPooling2D(2)(n_layer)

    return n_layer, skip_connection

def decoder_conv_block(l_layer, skip_layer, kernel_size = 3, n_filters = 32):
    """
    Function that implements a convolutional upsampling block.

    Arguments:
    l_layer -- Last tensor.
    skip_layer -- Last connection corespondent layer.
    n_filters -- Number of filters for the convolutional layers.

    Returns:
        n_layer -- Tensor output
    """ 

    n_layer = tf.keras.layers.Conv2DTranspose(n_filters,
                              kernel_size,
                              strides=2,
                              padding='same')(l_layer)

    # Merge the skip connection layer with current one
    merge = tf.keras.layers.concatenate([n_layer, skip_layer], axis=3)
    n_layer = tf.keras.layers.Conv2D(n_filters,
                                     kernel_size,
                                     activation='relu',
                                     padding='same',
                                     kernel_initializer='he_normal')(merge)
    
    n_layer = tf.keras.layers.Conv2D(n_filters,
                                     kernel_size,
                                     activation='relu',
                                     padding='same',
                                     kernel_initializer='he_normal')(merge)

    return n_layer

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """

    inputs = tf.keras.layers.Input(input_size)
    cblock1 = encoder_conv_block(inputs, n_filters)
    cblock2 = encoder_conv_block(cblock1[0], 2 * n_filters)
    cblock3 = encoder_conv_block(cblock2[0], 4 * n_filters)
    cblock4 = encoder_conv_block(cblock3[0], 8 * n_filters, dropout_prob=0.3)
    cblock5 = encoder_conv_block(cblock4[0], 16 * n_filters, dropout_prob=0.3, max_pooling=False)

    ublock6 = decoder_conv_block(cblock5[0], cblock4[1], n_filters = 8 * n_filters)
    ublock7 = decoder_conv_block(ublock6, cblock3[1], n_filters = 4 * n_filters)
    ublock8 = decoder_conv_block(ublock7, cblock2[1], n_filters = 2 * n_filters)
    ublock9 = decoder_conv_block(ublock8, cblock1[1], n_filters)
    
    conv9 = tf.keras.layers.Conv2D(n_filters, 
                                   kernel_size = 3,
                                   activation = 'relu',
                                   padding = 'same',
                                   kernel_initializer = 'he_normal')(ublock9)

    conv10 = tf.keras.layers.Conv2D(n_classes, kernel_size = 1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
