"""
工具类
create_time: 2018:/5/10
creator ：shura
"""
import numpy as np
from CRNN.keys import alphabet
import tensorflow as tf


class Util(object):
    def __init__(self):
        self.img_dir = "D:\\work\\tianchi\\train_set\\"
        self.IMAGE_HEIGHT = 128
        self.IMAGE_WIDTH = 128
        self.START_INDEX = 0
        self.encode_maps = {}
        self.decode_maps = {}
        for i, char in enumerate(alphabet, 1):
            self.encode_maps[char] = i
            self.decode_maps[i] = char

        SPACE_INDEX = 0
        SPACE_TOKEN = ''
        self.encode_maps[SPACE_TOKEN] = SPACE_INDEX
        self.decode_maps[SPACE_INDEX] = SPACE_TOKEN
        self.maxlength = len(self.decode_maps)

    # 序号转换为label
    def decode(self, x):
        return self.decode_maps[x]

    # label转换为序号
    def encode(self, x):
        return self.encode_maps[x]

    # one-hot编码
    def onehot(self, label_dicts):
        one_hot_labels = []
        for dic in label_dicts:
            one = [0.0 for _ in range(0, self.maxlength)]
            one[dic] = 1.0
            one_hot_labels.append(one)
        return one_hot_labels

    # 将数据集加载进来
    def load_labels(self, label_path):
        labels = list(open(label_path).readlines())
        # labels = "小帅是神！Yeah"
        labels = [s.strip() for s in labels]
        labels = [s.split() for s in labels]
        labels_seq = [x[0] for x in labels]
        labels_y = [x[1] for x in labels]
        # 将文字label转换为对应的索引 然后进行onehot编码
        labels_encode = [self.encode_maps[x[0]] for x in labels_y]
        labels_onehot = self.onehot(labels_encode)
        labels_seq = np.asarray(labels_seq, dtype=str)
        # print(labels_seq)
        # print(labels_onehot)
        # images = []
        # # 通过编号获取图像
        # for p in labels_seq:
        #     path = self.img_dir + p + '.jpg'
        #     img = Image.open(path)
        #     img = np.array(img)
        #     img = self.convert2gray(img)
        #     img = img.flatten() / 255
        #     img = tf.float32(img, 'float')
        #     images.append(img)
        # 返回训练集
        return labels_seq, labels_onehot

    # 图像灰化处理
    def convert2gray(self, img):
        # 将图片转化为灰色
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            # 上面的转法较快，正规转法如下
            # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    # 获取batch
    def get_batches(self, image, label, resize_w, resize_h, batch_size, capacity):
        # 将images and labels 的列表转换为 tensor 张量
        image = tf.cast(image, tf.string)
        label = np.asarray(label, dtype=float)
        label = tf.cast(label, tf.float32)
        queue = tf.train.slice_input_producer([image, label])
        label = queue[1]
        image_c = tf.read_file(self.img_dir + queue[0] + '.jpg')
        image = tf.image.decode_jpeg(image_c, channels=3)
        # resize
        image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        # (x - mean) / adjusted_stddev
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)
        images_batch = tf.cast(image_batch, tf.float32)
        labels_batch = tf.reshape(label_batch, shape=[batch_size, self.maxlength])
        return images_batch, labels_batch


if __name__ == '__main__':
    u = Util()
    # u.load_labels('labels.txt')
    images, labels = u.load_labels('labels.txt')
    batch_x, batch_y = u.get_batches(images, labels, 128, 128, 128, 100)
    print(batch_x)
    print(batch_y)