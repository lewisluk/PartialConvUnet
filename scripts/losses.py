import tensorflow as tf
from scripts.vgg_tf import Vgg16

def loss_hole(mask, y_true, y_pred):
    '''
    :param mask:    input_mask
    '''
    return tf.losses.absolute_difference((1 - mask) * y_true, (1 - mask) * y_pred)

def loss_valid(mask, y_true, y_pred):
    '''
    :param mask:    input_mask
    '''
    return tf.losses.absolute_difference(mask * y_true, mask * y_pred)

def ms_ssim_loss(predict, label):
    '''
    :param predict: generated image, value range 0~1, shape can be seen at tf.image.ssim_multiscale
    :param label:   label image, requirements stay the same
    :return:        multi-scale structural similarity loss
    '''
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(predict, label, max_val=1))

def tv_loss(image):
    '''
    :param image:   generated image, value range 0~1, shape can be seen at tf.image.total_variation
    :return:        total variation loss
    '''
    return tf.reduce_mean(tf.image.total_variation(image))

def perceptual_loss(vgg16, fake, label, y_comp):
    batch = tf.concat(values=[fake, label, y_comp], axis=0)
    vgg16.build(batch)
    vgg_out = [vgg16.conv1_2, vgg16.conv2_2, vgg16.conv3_3, vgg16.conv4_3, vgg16.conv5_3]
    scaled_vgg_loss1 = tf.reduce_mean(
        [(1.0 / p[0].get_shape().as_list()[-1]) * tf.reduce_mean(tf.abs(p[0] - p[1])) for p in vgg_out])

    scaled_vgg_loss2 = tf.reduce_mean(
        [(1.0 / p[0].get_shape().as_list()[-1]) * tf.reduce_mean(tf.abs(p[2] - p[1])) for p in vgg_out])

    scaled_vgg_loss = scaled_vgg_loss1 + scaled_vgg_loss2
    return scaled_vgg_loss

def get_gram(layer_out):
    """
    This function computes the style matrix, which essentially computes
    how correlated the activations of a given filter to all the other filers.
    Therefore, if there are C channels, the matrix will be of size C x C
    """
    n_channels = layer_out.get_shape().as_list()[-1]
    unwrapped_out = tf.reshape(layer_out, [-1, n_channels])
    style_matrix = tf.matmul(unwrapped_out, unwrapped_out, transpose_a=True)
    return style_matrix

def gram_loss(vgg16):
    '''
    :param vgg16:   loaded vgg16 model
    :return:        gram loss based on vgg16
    '''
    vgg_out = [vgg16.pool1, vgg16.pool2, vgg16.pool3, vgg16.pool4, vgg16.pool5]
    fake_gram = [get_gram(p[0]) for p in vgg_out]
    label_gram = [get_gram(p[1]) for p in vgg_out]
    y_comp_gram = [get_gram(p[2]) for p in vgg_out]

    loss1 = 1e-15 * tf.reduce_mean(
        [(1.0 / v[0].get_shape().as_list()[-1]) * tf.reduce_mean((c - g) ** 2) for c, g, v in
         zip(fake_gram, label_gram, vgg_out)], 0)

    loss2 = 1e-15 * tf.reduce_mean(
        [(1.0 / v[0].get_shape().as_list()[-1]) * tf.reduce_mean((c - g) ** 2) for c, g, v in
         zip(y_comp_gram, label_gram, vgg_out)], 0)

    return loss1 + loss2

def vgg_loss(predict, label, mask, percep_weight, gram_weight):
    y_comp = mask * label + (1 - mask) * predict

    fake = tf.image.resize_image_with_crop_or_pad(image=predict, target_height=224, target_width=224)
    label = tf.image.resize_image_with_crop_or_pad(image=label, target_height=224, target_width=224)
    y_comp = tf.image.resize_image_with_crop_or_pad(image=y_comp, target_height=224, target_width=224)

    vgg16 = Vgg16('vgg16.npy')
    percep_loss = perceptual_loss(vgg16, fake, label, y_comp)
    gram_l = gram_loss(vgg16)
    return percep_weight * percep_loss, gram_weight * gram_l

