import tensorflow as tf
from tensorflow.contrib import rnn

class CRNNNet():

    def __init__(self, phase, hidden_nums, layers_nums, seq_length, num_classes):
        """
        参数初始化
        :param phase:
        :param hidden_nums:
        :param layers_nums:
        :param seq_length:
        :param num_classes:
        """
        self.phase = phase
        self.hidden_nums = hidden_nums
        self.layers_nums = layers_nums
        self.seq_length = seq_length
        self.num_classes = num_classes

    def cnn_VGG(self,inputdata,isTraining):
        """
        建立CRNN网络
        """
        conv1 = self.conv_op(inputdata=inputdata, out_dims=64, name='conv1')        # batch*16*50*64
        conv2 = self.conv_op(inputdata=conv1, out_dims=128, name='conv2')           # batch*8*25*128
        # 此处使用自定义的卷积核与步长所以未使用封装固定值的2*2大小
        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False,
                            name='conv3')                                           # batch*8*25*256
        relu3 = tf.nn.relu(conv3)                                                   # batch*8*25*256
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False,
                            name='conv4')                                           # batch*8*25*256
        relu4 = tf.nn.relu(conv4)                                                   # batch*8*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1],
                                    padding='VALID')                                # batch*4*25*256
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False,
                            name='conv5')                                           # batch*4*25*512
        relu5 = tf.nn.relu(conv5)                                                   # batch*4*25*512
        if isTraining == 'train':
            bn5 = tf.contrib.layers.batch_norm(relu5, scale=True, is_training=True, updates_collections=None)
        else:
            bn5 = tf.contrib.layers.batch_norm(relu5, scale=True, is_training=False, updates_collections=None)  # batch*4*25*512
        conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3, stride=1, use_bias=False,
                            name='conv6')                                           # batch*4*25*512
        relu6 = tf.nn.relu(conv6)                                                   # batch*4*25*512
        if isTraining == 'train':
            bn6 = tf.contrib.layers.batch_norm(relu6, scale=True, is_training=True, updates_collections=None)
        else:
            bn6 = tf.contrib.layers.batch_norm(relu6, scale=True, is_training=False, updates_collections=None)  # batch*4*25*512
        max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1], stride=[2, 1])                           # batch*2*25*512
        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7')  # batch*1*25*512
        relu7 = tf.nn.relu(conv7)                                                   # batch*1*25*512
        return relu7

    def map_to_sequence(self, inputdata):
        """
        将网络提取出的featuremap转换为lstm层可使用的sequence
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input=inputdata, axis=1)

    def sequence_label(self, inputdata,isTraining):
        """
        实现网络序列的label
        :param inputdata:
        :return:
        """
        with tf.variable_scope('LSTMLayers'):
            # 构造 lstm rcnn layer
            # 前向 lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_nums, self.hidden_nums]]
            # 后向 direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_nums, self.hidden_nums]]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                         dtype=tf.float32)

            if isTraining == 'train':
                stack_lstm_layer = tf.nn.dropout(stack_lstm_layer, keep_prob=0.5, noise_shape=None, name=None)
            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()              # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])           # [batch x width, 2*n_hidden]
            w = tf.Variable(tf.truncated_normal([hidden_nums, self.num_classes], stddev=0.1), name="w")
            # 仿射投影到num_class上
            # 未进入到softmax之前的概率
            logits = tf.matmul(rnn_reshaped, w)
            logits = tf.reshape(logits, [batch_s, -1, self.num_classes])
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')
            # 交换batch和轴 转置
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]
        return rnn_out, raw_pred

    def conv_op(self, inputdata,out_dims,name=None):
        """
        封装一个固定参数的卷积+池化操作
        :param inputdata:
        :param out_dims:
        :param name:
        :return:
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False,
                           name=name)
        relu = tf.nn.relu(conv,name=name)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def conv2d(self,inputdata, out_channel, kernel_size, padding='SAME', stride=1, w_init=None, b_init=None,
               nl=tf.identity, split=1, use_bias=True, data_format='NHWC', name=None):
        """
        定义卷积操作
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param nl: a tensorflow identify function
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
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

    def maxpooling(self, inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        """
        定义池化操作
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

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    def build_CRNN(self, inputdata):
        """
        建立CRNN网络
        :param inputdata:
        :return:
        """
        # 通过VGG提取图片特征
        cnn_out = self.cnn_VGG(inputdata=inputdata)

        # 然后将特征映射到序列中
        sequence = self.map_to_sequence(inputdata=cnn_out)

        # 然后通过lstm得到对应的label
        net_out, raw_pred = self.sequence_label(inputdata=sequence)
        return net_out