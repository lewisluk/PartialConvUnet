from scripts.network_ops import *
from scripts.losses import *
from scripts.data_io import io_handler
import tensorflow.contrib.slim as slim, os, numpy as np

class PartialConvUnet():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.input = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='inpaiting_input')
        self.input_mask = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='input_mask')
        self.label = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='y_label')
        self.data_dir = '/data/PhotoInpainting/scenery512'
        self.result_dir = 'result'
        self.noisy_label = tf.placeholder(tf.float32, shape=None)
        self.batch_size = 12
        self.learningrate = 5e-4
        self.epochs = 1000
        self.clip = 0.1
        self.valid_weight = 10
        self.hole_weight = 60
        self.ssim_loss_weight = 1
        self.gram_weight = 1
        self.percep_weight = 1
        self.tv_weight = 1e-5

    def restore(self, ckpt_path=None):
        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
        else:
            self.saver.restore(self.sess, self.result_dir + '/model.ckpt')

    def build_G(self, inputs):
        with tf.variable_scope("generator"):
            e_conv1, self.e_mask1 = partial_conv(inputs, self.input_mask, 64, kernel_size=7,
                                            ln=False, scope='en1')
            e_conv2, self.e_mask2 = partial_conv(e_conv1, self.e_mask1, 128, kernel_size=5, scope='en2')
            e_conv3, self.e_mask3 = partial_conv(e_conv2, self.e_mask2, 256, kernel_size=5, scope='en3')
            e_conv4, self.e_mask4 = partial_conv(e_conv3, self.e_mask3, 512, scope='en4')
            e_conv5, self.e_mask5 = partial_conv(e_conv4, self.e_mask4, 512, scope='en5')
            e_conv6, self.e_mask6 = partial_conv(e_conv5, self.e_mask5, 512, scope='en6')
            e_conv7, self.e_mask7 = partial_conv(e_conv6, self.e_mask6, 512, scope='en7')

            d_conv8, self.d_mask8 = decoder_layer(feature_in=e_conv7, mask_in=self.e_mask7,
                                               e_conv=e_conv6, e_mask=self.e_mask6,
                                               filters=512, scope='de10')
            d_conv9, self.d_mask9 = decoder_layer(feature_in=d_conv8, mask_in=self.d_mask8,
                                               e_conv=e_conv5, e_mask=self.e_mask5,
                                               filters=512, scope='de11')
            d_conv10, self.d_mask10 = decoder_layer(feature_in=d_conv9, mask_in=self.d_mask9,
                                               e_conv=e_conv4, e_mask=self.e_mask4,
                                               filters=512, scope='de12')
            d_conv11, self.d_mask11 = decoder_layer(feature_in=d_conv10, mask_in=self.d_mask10,
                                               e_conv=e_conv3, e_mask=self.e_mask3,
                                               filters=256, scope='de13')
            d_conv12, self.d_mask12 = decoder_layer(feature_in=d_conv11, mask_in=self.d_mask11,
                                               e_conv=e_conv2, e_mask=self.e_mask2,
                                               filters=128, scope='de14')
            d_conv13, self.d_mask13 = decoder_layer(feature_in=d_conv12, mask_in=self.d_mask12,
                                               e_conv=e_conv1, e_mask=self.e_mask1,
                                               filters=64, scope='de15')
            d_conv14, self.mask_out = decoder_layer(feature_in=d_conv13, mask_in=self.d_mask13,
                                               e_conv=inputs, e_mask=self.input_mask,
                                               filters=3, ln=False, scope='de16')

            out = slim.conv2d(d_conv14, 3, [1, 1], activation_fn=None, scope='inpainting_out')
            return out

    def build_D(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("discriminator", reuse=reuse):
            h = slim.conv2d(x, 16, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="1_d")
            h = slim.conv2d(h, 32, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="2_d")
            h = slim.conv2d(h, 64, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="3_d")
            h = slim.conv2d(h, 128, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="4_d")
            h = slim.conv2d(h, 128, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="5_d")
            h = slim.conv2d(h, 128, stride=2, kernel_size=3, activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.instance_norm, scope="6_d")

            reshaped = tf.reshape(h, [-1, 16 * 16])
            fc = slim.fully_connected(reshaped, 1, activation_fn=None, scope='d_fc')
        return fc

    def init_network(self):
        '''
        GAN related loss was implemented as LSGAN
        D clip and optimizers are implemented as WGAN
        '''
        # loss
        self.predict = self.build_G(self.input)

        d_out_real = self.build_D(self.label)
        d_out_fake = self.build_D(self.predict)

        d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(d_out_real, tf.ones_like(d_out_real)))
        d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(d_out_fake, tf.zeros_like(d_out_fake)))

        self.D_loss = (d_loss_real + d_loss_fake) / 2
        self.G_loss = tf.reduce_mean(tf.losses.mean_squared_error(d_out_fake, tf.ones_like(d_out_fake)))

        self.l1_valid = self.valid_weight * loss_valid(self.input_mask, self.label, self.predict)
        self.l1_hole = self.hole_weight * loss_hole(self.input_mask, self.label, self.predict)
        self.tv_loss = self.tv_weight * tv_loss(self.predict)
        self.ssim_loss = self.ssim_loss_weight * ms_ssim_loss(self.predict, self.label)
        self.percep_loss, self.gram_loss = vgg_loss(self.predict, self.label, self.input_mask, self.percep_weight,
                                                    self.gram_weight)
        self.loss = self.l1_valid + self.l1_hole +self.ssim_loss + \
                    self.G_loss + self.percep_loss + self.gram_loss + self.tv_loss

        self.PSNR = tf.image.psnr(self.predict, self.label, 1)
        self.SSIM = tf.image.ssim(self.predict, self.label, 1)

        var_g = [v for v in tf.trainable_variables() if 'generator' in v.name]
        var_d = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learningrate).minimize(self.loss, var_list=var_g)

        # apply gradient clipping to d_opt
        d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningrate)
        gvs = d_optimizer.compute_gradients(self.D_loss, var_list=var_d)
        capped_gvs = [(tf.clip_by_value(grad, -self.clip, self.clip), var) for grad, var in gvs]
        self.d_opt = d_optimizer.apply_gradients(capped_gvs)

        self.saver = tf.train.Saver(max_to_keep=self.epochs)
        self.sess.run(tf.global_variables_initializer())
        self.io_handler = io_handler(self.data_dir, self.result_dir, self.batch_size)

    def trainer(self):
        for epoch in range(self.epochs):
            cnt = 0
            train_psnr, train_loss, train_ssim, train_l1_hole,\
            train_l1_valid, train_ssim_loss, train_perc,\
            train_gram, train_Gloss, train_Dloss, train_gram, train_tv = \
            [], [], [], [], [], [], [], [], [], [], [], []
            iters, mod = divmod(len(self.io_handler.train_names), self.batch_size)
            for j in range(iters):
                input_batch, mask_batch, label_batch = self.io_handler.load_batch(j)
                _, _, loss, g_loss, d_loss, psnr, ssim, \
                l1_valid, l1_hole, ssim_loss, perc, gram, tv = self.sess.run(
                    [self.opt,
                     self.d_opt,
                     self.loss,
                     self.G_loss,
                     self.D_loss,
                     self.PSNR,
                     self.SSIM,
                     self.l1_valid,
                     self.l1_hole,
                     self.ssim_loss,
                     self.percep_loss,
                     self.gram_loss,
                     self.tv_loss],
                    feed_dict={self.input: input_batch,
                               self.input_mask: mask_batch,
                               self.label: label_batch})

                train_loss.append(loss)
                train_Gloss.append(g_loss)
                train_Dloss.append(d_loss)
                train_psnr.append(psnr)
                train_ssim.append(ssim)
                train_l1_valid.append(l1_valid)
                train_l1_hole.append(l1_hole)
                train_ssim_loss.append(ssim_loss)
                train_perc.append(perc)
                train_gram.append(gram)
                train_tv.append(tv)
                cnt += 1

            print(
                'Ep:{}, Iter:{}, PSNR:{:.4f}, SSIM:{:.4f}, '
                'Loss:{:.4f}, l1_valid:{:.4f}, l1_hole:{:.4f}, ssim_L:{:.4f}, '
                'perc:{:.4f}, gram:{:.4f}, tv:{:.4f}, Gloss:{:.4f}, Dloss:{:.4f}'.format(
                    epoch + 1,
                    cnt,
                    np.mean(train_psnr),
                    np.mean(train_ssim),
                    np.mean(train_loss),
                    np.mean(train_l1_valid),
                    np.mean(train_l1_hole),
                    np.mean(train_ssim_loss),
                    np.mean(train_perc),
                    np.mean(train_gram),
                    np.mean(train_tv),
                    np.mean(train_Gloss),
                    np.mean(train_Dloss)
                    ))

            if not os.path.exists("%s/%03d" % (self.result_dir, epoch + 1)):
                os.makedirs("%s/%03d" % (self.result_dir, epoch + 1))
            val_psnr, val_ssim, val_loss = [], [], []

            iters, mod = divmod(len(self.io_handler.val_names), self.batch_size)

            for j in range(iters):
                input_batch, mask_batch, label_batch = self.io_handler.load_batch(j, training=False)
                # intermediate_masks = [self.e_mask1, self.e_mask2, self.e_mask3, self.e_mask4,
                #                       self.e_mask5, self.e_mask6, self.e_mask7, self.d_mask8,
                #                       self.d_mask9, self.d_mask10, self.d_mask11, self.d_mask12,
                #                       self.d_mask13, self.mask_out]
                val_output, temp_psnr, temp_ssim, temp_loss = self.sess.run([
                    self.predict, self.PSNR, self.SSIM, self.loss], feed_dict={
                    self.input: input_batch, self.input_mask: mask_batch,
                    self.label: label_batch})

                self.io_handler.save_batch(val_output, epoch, j)
                val_psnr.append(temp_psnr)
                val_ssim.append(temp_ssim)
                val_loss.append(temp_loss)
            print('Validation for Epoch{:d}: PSNR:{:.4f}, SSIM:{:.4f}, Loss:{:.4f}'.format(
                epoch + 1,
                np.mean(val_psnr),
                np.mean(val_ssim),
                np.mean(val_loss)
            ))

            self.saver.save(self.sess, self.result_dir + '/{:03d}/model.ckpt'.format(epoch + 1))