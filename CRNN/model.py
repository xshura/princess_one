"""
模型类
CNN + LSTM + CTC
create_time: 2018/5/10
creator ：shura
"""
import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, hidden_nums, seq_length, num_classes):
        """
        初始化参数：
        :param hidden_nums:  lstm隐藏单元
        :param seq_length:   序列长度
        :param num_classes:  label个数
        """
        self.seq_length = seq_length
        self.num_class = num_classes
        self.hidden_nums = hidden_nums

    def conv_op(self, inputdata,out_dims,name=None):
        """
        封装一个固定参数的卷积+池化操作
        :param inputdata:
        :param out_dims:
        :param name:
        :return:
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
        relu = tf.nn.relu(conv,name=name)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def maxpooling(self, inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        """
        定义池化操作 可能传入参数为list所以此处要进行处理
        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()
        if stride is None:
            stride = kernel_size
        # 判断输入格式是否批量传入
        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]
        # 判断是否批量传输数据
        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]
        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    def conv2d(self,inputdata, out_channel, kernel_size, padding='SAME', stride=1, w_init=None, b_init=None,
               nl=tf.identity, split=1, use_bias=True, data_format='NHWC', name=None):
        """
        定义卷积操作 判断是否使用偏执
        :param name: op name
        :param inputdata: 4维的tensorflow张量 [batchsize, w, h ,c]
        :param out_channel: channel 数量
        :param kernel_size: 卷积核大小
        :param padding: 'VALID' or 'SAME' padding模式
        :param stride: 卷积步长
        :param w_init: 权重是否初始化
        :param b_init: i是否初始化偏置
        :param nl: 验证函数
        :param split: split channels as used in Alexnet mainly group for GPU memory save.要是在alexnet中使用GPU保存的时候需要将chennel分开
        :param use_bias:  是否使用偏置
        :param data_format: 设定数据格式
        :return: tf.Tensor named ``output``返回输出结果
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0
            padding = padding.upper()
            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]
            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]
            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()
            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None
            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)
            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)
            res = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name=name)
        return res

    def cnn_VGG(self, input_data):
        """
        采用vgg作为卷积网络
        :param input_data:
        :return: output tensor
        """
        conv_1 = self.conv_op(input_data, out_dims=64, name='conv1')                  # kernel 3,3 relu 64
        conv_2 = self.conv_op(conv_1, out_dims=128, name='conv2')                     # kernel 3,3 relu 128
        conv_3 = self.conv2d(inputdata=conv_2, out_channel=256, kernel_size=3, stride=1,
                             use_bias=False, name='conv3')                            # kernel 3,3 relu 256
        relu_3 = tf.nn.relu(conv_3)
        conv_4 = self.conv2d(inputdata=relu_3, out_channel=256, kernel_size=3, stride=1, use_bias=False,
                             name='conv4')                                            # kernel 3,3 relu 256
        relu_4 = tf.nn.relu(conv_4)
        maxpool_4 = self.maxpooling(relu_4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')
        conv_5 = self.conv2d(inputdata=maxpool_4, out_channel=512, kernel_size=3, stride=1, use_bias=False,
                             name='conv5')                                            # kernel 3,3 relu 512
        relu_5 = tf.nn.relu(conv_5)
        batch_normal_5 = tf.contrib.layers.batch_norm(relu_5, scale=True, is_training=True, updates_collections=None)
        conv_6 = self.conv2d(inputdata=batch_normal_5, out_channel=512, kernel_size=3, stride=1, use_bias=False,
                             name='conv6')                                            # kernel 3,3 relu 512
        relu_6 = tf.nn.relu(conv_6)
        batch_normal_6 = tf.contrib.layers.batch_norm(relu_6, scale=True, is_training=True, updates_collections=None)
        maxpool_6 = self.maxpooling(inputdata=batch_normal_6, kernel_size=[2, 1], stride=[2, 1])
        conv_7 = self.conv2d(inputdata=maxpool_6, out_channel=512, kernel_size=2, stride=1, use_bias=False,
                             name='conv7')                                            # kernel 2,2 relu 512
        relu_7 = tf.nn.relu(conv_7)
        return relu_7

    def map_to_sequence(self, inputdata, batch_size):
        """
        将经过cnn提取出来的featuremap转换为可以进入lstm的sequence
        :param inputdata:
        :return: output tensor
        """
        reshaped_cnn_output = tf.reshape(inputdata, [batch_size, -1, 512])
        shape = inputdata.get_shape().as_list()[1]
        return reshaped_cnn_output, shape

    def BidirectionalLSTM(self, inputdata):
        """
        创建双向lstm网络
        :param inputdata:
        :return:
        """
        # 多层网络 论文实现为两层256个隐层单元
        # 前向cell
        fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_nums, self.hidden_nums]]
        # 后向 direction cells
        bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_nums, self.hidden_nums]]

        stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                     sequence_length=self.seq_length, dtype=tf.float32)
        # 添加dropout层
        stack_lstm_layer = tf.nn.dropout(stack_lstm_layer, keep_prob=0.5, noise_shape=None, name=None)
        [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()
        outputs = tf.reshape(stack_lstm_layer, [-1, hidden_nums])
        weights = tf.get_variable(name='W_out',
                                  shape=[hidden_nums, self.num_class],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
        bias = tf.get_variable(name='b_out',
                               shape=[self.num_class],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer())
        # 仿射投影到num_class上
        # 未进入到softmax之前的概率
        logits = tf.matmul(outputs, weights) + bias
        logits = tf.reshape(logits, [batch_s, -1, self.num_class])
        # 交换batch和轴 转置
        raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')
        logits = tf.transpose(logits, (1, 0, 2))
        return logits, raw_pred

    def build_model(self, inputdata):
        """
        组合模型 CNN + LSTM 构成CRNN
        :param inputdata:
        :return:
        """
        # 通过VGG提取图片特征
        cnn_out = self.cnn_VGG(inputdata)

        # 然后将特征映射到序列中
        sequence, _ = self.map_to_sequence(inputdata=cnn_out, batch_size=128)

        # 然后通过lstm得到对应的输出
        net_out, raw_pred = self.BidirectionalLSTM(inputdata=sequence)

        return net_out, raw_pred




