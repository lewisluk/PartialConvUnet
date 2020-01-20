from sklearn.model_selection import train_test_split
import os, numpy as np, cv2, shutil
from scripts.mask_generator import MaskGenerator

class io_handler():
    def __init__(self, data_dir, result_dir, batch_size):
        self.result_dir = result_dir
        self.data_dir = data_dir
        self.train_names, self.val_names = self.process_data(data_dir, result_dir)
        self.G_mask = MaskGenerator(height=512, width=512, rand_seed=1)
        self.batch_size = batch_size

    def process_data(self, data_dir, result_dir):
        img_names = os.listdir(data_dir)
        train_names, val_names = train_test_split(img_names, test_size=0.05, random_state=1)
        if not os.path.exists(result_dir + '/val_label'):
            os.mkdir(result_dir + '/val_label')
            for i in val_names:
                shutil.copy(os.path.join(data_dir, i),
                            os.path.join(result_dir, 'val_label', i))
        return train_names, val_names

    def preprocess(self, img_path):    # 训练前数据预处理
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        mask = self.G_mask.sample()
        condition = np.equal(mask, np.ones_like(mask))
        img_with_holes = np.where(condition, img, np.ones_like(mask) * 255.)/255.

        raw_img = img / 255.

        mask = mask.astype(np.float32)
        return img_with_holes, mask, raw_img

    def postprocess(self, net_out):    # 数据后处理
        net_out = np.clip(net_out, 0.0, 1.0)
        net_out = net_out * 255
        out = net_out.astype(np.uint8)
        return out

    def load_batch(self, iter, training=True):
        input_batch, mask_batch, label_batch = [], [], []
        names = self.train_names if training else self.val_names
        for i in range(self.batch_size):
            current_img_path = os.path.join(self.data_dir, names[iter + i])
            img_with_holes, mask, raw_img = self.preprocess(current_img_path)
            input_batch.append(img_with_holes)
            mask_batch.append(mask)
            label_batch.append(raw_img)
        input_batch = np.array(input_batch)
        self.mask_batch = np.array(mask_batch)  # for saving
        label_batch = np.array(label_batch)
        return input_batch, self.mask_batch, label_batch

    def save_batch(self, pred_batch, epoch, iter):
        for i in range(self.batch_size):
            current_pred = self.postprocess(pred_batch[i])
            current_mask = self.postprocess(self.mask_batch[i])
            cv2.imwrite(self.result_dir +
                        '/{:03d}/{:s}_mask.png'.format(
                            epoch + 1,
                            self.val_names[iter+i].replace('.jpg', '')),
                        current_mask)
            cv2.imwrite(self.result_dir +
                        '/{:03d}/{:s}'.format(
                            epoch + 1,
                            self.val_names[iter+i]),
                        current_pred)
            # for j in range(len(masks_batch)):
            #     current_intermediate_mask = self.postprocess(masks_batch[j][i, :, :, 0])
            #     cv2.imwrite(self.result_dir +
            #             '/{:03d}/{:s}_inter_mask_{:d}.png'.format(
            #                 epoch + 1,
            #                 self.val_names[iter+i].replace('.jpg', ''),
            #             j),
            #             current_intermediate_mask)