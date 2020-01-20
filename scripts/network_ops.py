import tensorflow as tf
from tensorflow.contrib import slim

def decoder_layer(feature_in, mask_in, e_conv, e_mask, filters, scope, ln=True):

    feature_in = tf.image.resize_bilinear(feature_in, tf.shape(e_conv)[1:3])
    mask_in = tf.image.resize_bilinear(mask_in, tf.shape(e_mask)[1:3])

    # concat feature and mask with previous layers
    concat_feature = tf.concat([e_conv, feature_in], axis=-1)
    concat_mask = tf.concat([e_mask, mask_in], axis=-1)

    # fuse feature and mask by partial conv
    feature_out, mask_out = partial_conv(concat_feature,
                                         concat_mask,
                                         filters,
                                         strides=1,
                                         ln=ln,
                                         scope=scope)
    return feature_out, mask_out

def partial_conv(feature, mask, num_outputs, kernel_size=3, strides=2, ln=True, scope=None):
    '''
    Partial convolutional layer to downsample both feature and mask
    '''

    # apply zero padding to both feature and mask
    pconv_padding = [[0, 0],
                     [int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)],
                     [int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)],
                     [0, 0]]
    feature = tf.pad(feature, pconv_padding, "CONSTANT")
    mask = tf.pad(mask, pconv_padding, "CONSTANT")

    # apply conv to both feature and mask
    feature_output = tf.layers.conv2d(inputs=feature*mask,
                                      filters=num_outputs,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=None,
                                      use_bias=False,
                                      name=scope+'_feature')

    mask_output = tf.layers.conv2d(inputs=mask,
                                   filters=num_outputs,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   activation=None,
                                   kernel_initializer=tf.ones_initializer,
                                   use_bias=False,
                                   trainable=False,
                                   name=scope+'_mask')

    # Calculate the mask ratio on each pixel in the output mask
    window_size = kernel_size * kernel_size * feature.shape._dims[3]._value
    mask_ratio = tf.divide(window_size, (mask_output + 1e-8))

    # Clip output to be between 0 and 1
    mask_output = tf.clip_by_value(mask_output, 0, 1)

    # Remove ratio values where there are holes
    mask_ratio = mask_ratio * mask_output

    # Normalize feature output
    feature_output = feature_output * mask_ratio

    if ln:
        feature_output = slim.layer_norm(feature_output, scope=scope+'_ln')

    feature_output = tf.nn.selu(feature_output)
    return feature_output, mask_output
