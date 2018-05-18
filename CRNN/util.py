"""
工具类
create_time: 2018/5/10
creator ：shura
"""
import random

import cv2
import numpy as np

from CRNN.keys import alphabet
import tensorflow as tf


class Util(object):
    def __init__(self):
        self.img_dir = "D:\\work\\tianchi\\train_set\\"
        self.IMAGE_HEIGHT = 64
        self.IMAGE_WIDTH = 256
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

        images, labels = self.load_labels('labels.txt')
        self.images = images
        self.labels = labels

    # 序号转换为label
    def decode(self, x):
        return self.decode_maps[x]

    # label转换为序号
    def encode(self, x):
        return self.encode_maps[x]

    # label的序列转为稀疏矩阵
    def seq_to_sparseTensor(self, seq):
        """
        indices:二维int64的矩阵，代表非0的坐标点
        values:二维tensor，代表indice位置的数据值
        dense_shape:一维，代表稀疏矩阵的大小
        仍然拿刚才的两个串"12"和"1"做例子，转成的稀疏矩阵应该是
        indecs = [[0,0],[0,1],[1,0]]
        values = [1,2,1]
        dense_shape = [2,2] (两个数字串，最大长度为2)
        :param seq:
        :return:
        """
        indices = []
        values = []
        for n, seq in enumerate(seq):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(seq), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
        return indices, values, shape

    # 将稀疏矩阵转换为label序列
    def sparseTensor_to_seq(self, sparse_tensor):
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i: decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)
        result = []
        for index in decoded_indexes: result.append(self.decode_a_seq(index, sparse_tensor))
        return result

    def decode_a_seq(self, indexes, spars_tensor):
        decoded = []
        for m in indexes:
            str = self.decode_maps[spars_tensor[1][m]]
            decoded.append(str)
        return decoded

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
        labels_seq = []
        labels_y = []
        for x in labels:
            # 如果数据畸形就丢弃掉
            if len(x) < 2: continue
            labels_seq.append(x[0])
            l = x[1]
            # 判断如果大于两位就把后面的连接在一起作为label
            if len(x) > 2:
                for i in range(2,len(x)):
                    l += i
            labels_y.append(l)
        # 将文字label转换为对应的索引 然后进行onehot编码
        labels_encode = []
        for label in labels_y:
            l = [int(self.encode_maps[x]) for x in label]
            labels_encode.append(l)
        # 因为ctc要求转化为稀疏矩阵故此处不适用onehot
        # labels_onehot = self.onehot(labels_encode)
        # label_path = [ self.img_dir + p + '.jpg' for p in labels_seq]
        # label_path = np.asarray(label_path, dtype=str)
        labels_target = labels_encode
        # print(labels_seq)
        # print(labels_onehot)
        images = []
        # 通过编号获取图像
        for p in labels_seq:
            path = self.img_dir + p + '.jpg'
            img = cv2.imread(path)
            if img.shape[1] < img.shape[0]:
                img = np.rot90(img)
            img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            images.append(img)
        # 返回训练集
        return images, labels_target

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
        targets = [np.asarray(i) for i in label]
        # targets转成稀疏矩阵
        sparse_targets, _, _ = self.seq_to_sparseTensor(targets)

        # 通过tensorflow内置的方法构建数据集
        queue = tf.train.slice_input_producer([image, label])
        target = queue[1]
        image_c = tf.read_file(self.img_dir + queue[0] + '.jpg')
        image = tf.image.decode_jpeg(image_c, channels=3)
        # resize
        image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        # (x - mean) / adjusted_stddev 标准差
        image = tf.image.per_image_standardization(image)
        # 获取tensorflow处理后的batch
        image_batch, label_batch = tf.train.batch([image, target],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)
        images_batch = tf.cast(image_batch, tf.float32)
        labels_batch = tf.reshape(label_batch, shape=[batch_size, 1])
        return images_batch, labels_batch

    # 获取下一组batch的训练数据
    def get_next_batch(self, batch_size):
        inputs = np.zeros(shape=[batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3], dtype=float)
        # if (batch_size+self.START_INDEX) >= len(self.labels):
        #     inputs = np.zeros(shape=[len(self.labels)-self.START_INDEX, 128, 128, 3], dtype=float)
        targets = []
        # 改为有放回的取出样本
        # for i in range(batch_size):
        #     j = random.randint(0, len(self.images)-1)
        #     inputs[i, :] = self.images[j].reshape((128, 128, 3))
        #     targets.append(self.labels[j])
        for i in range(batch_size):
            if (self.START_INDEX + i) >= len(self.labels):
                self.START_INDEX = 0
            inputs[i, :] = self.images[self.START_INDEX+1].reshape((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)).transpose((1, 0, 2))
            targets.append(self.labels[self.START_INDEX+1])
            self.START_INDEX += 1
        labels = [np.asarray(i) for i in targets]
        # targets转成稀疏矩阵
        sparse_targets = self.seq_to_sparseTensor(labels)
        # (batch_size,) sequence_length值都是256，最大划分列数
        seq_len = np.ones(inputs.shape[0]) * self.IMAGE_WIDTH
        return inputs, sparse_targets, seq_len


def sparseTensor_to_seq(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i: decoded_indexes.append(current_seq)
        current_i = i
        current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes: result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        u = Util()
        str = u.decode_maps[spars_tensor[1][m]]
        decoded.append(str)
    return decoded


if __name__ == '__main__':
    u = Util()
    # u.load_labels('labels.txt')
    batch_x, batch_y, seq_len = u.get_next_batch(batch_size=128)
    print(batch_x[0])
    print(batch_y[0])