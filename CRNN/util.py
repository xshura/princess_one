"""
模型工具类
"""
import numpy as np
from CRNN.keys import alphabet
from PIL import Image
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
            one = [0.0 for _ in range(0, self.maxlength + 1)]
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
        labels_onehot = np.asarray(labels_onehot, dtype=float)
        labels_onehot = tf.cast(labels_onehot, tf.float32)
        # print(labels_seq)
        # print(labels_onehot)
        images = []
        # 通过编号获取图像
        for p in labels_seq:
            path = self.img_dir + p + '.jpg'
            img = Image.open(path)
            img = np.array(img)
            img = self.convert2gray(img)
            img = img.flatten() / 255
            img = tf.float32(img, 'float')
            images.append(img)
        # 返回训练集
        return images, labels_onehot

    # 获取下一个batch
    def get_next_batch(self, batch_size):
        batch_x = np.zeros([batch_size, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, self.maxlength+1])
        train_X, train_Y = self.load_labels('labels.txt')
        end_index = self.START_INDEX + batch_size
        if end_index >= len(train_X): end_index = len(train_X)
        for i in range(0, batch_size):
            if (self.START_INDEX+i) >= self.maxlength: break
            image, label = train_X[self.START_INDEX+i], train_Y[end_index+i]
            batch_x[i, :] = image
            batch_y[i, :] = label
        self.START_INDEX = end_index
        return batch_x, batch_y

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


if __name__ == '__main__':
    u = Util()
    # u.load_labels('labels.txt')
    batch_x, batch_y = u.get_next_batch(3)
    print(batch_x)
    print(batch_y)