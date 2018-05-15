import tensorflow as tf
import time

from CRNN.model import Model
from CRNN.util import Util

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 1
EPOCH_NUM = 100000
# 初始化学习率
init_learning_rate = 0.01
# 衰变学习率参数
decay_steps = 10000
decay_rate = 0.98
# Adam优化器的参数
beta1 = 0.9
beta2 = 0.999
REPORT_STEPS = 100
BATCHES = 10


def ctc_loss(logits, labels, seq_len):
    """
    因为无法固定输入labels的长度故此处采用CTC_loss
    :param logits:
    :param labels:
    :param seq_len:
    :return:
    """
    loss = tf.nn.ctc_loss(labels=labels,
                          inputs=logits,
                          sequence_length=seq_len)
    cost = tf.reduce_mean(loss)
    return cost


def optimizer_op(loss):
    """
    衰变学习率 优化器
    :param loss: 传入ctc_loss 函数
    :return:
    """
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(init_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2).minimize(loss, global_step=global_step)
    return optimizer


def train():
    u = Util()
    # image输入
    inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    # SparseTensor CTC需要的参数
    labels = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    # 获取模型
    model = Model(hidden_nums=256, seq_length=seq_len, num_classes=u.maxlength)
    logits, raw_pred = model.build_model(inputs)

    # 优化器
    loss = ctc_loss(labels=labels, logits=logits, seq_len=seq_len)
    optimizer = optimizer_op(loss)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)
        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for i in range(EPOCH_NUM):
            train_cost = train_ler = 0
            train_inputs, train_targets, train_seq_len = u.get_next_batch(18)
            val_feed = {inputs: train_inputs, labels: train_targets, seq_len: train_seq_len}
            val_cost, val_ler, steps, _ = sess.run([loss, acc, global_step, optimizer], feed_dict=val_feed)
            print("Epoch.......", i)
            if i % 10 == 0:
                saver.save(sess, "./models/crnn.ckpt", global_step=steps)
            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}"
            print(log.format(i + 1, EPOCH_NUM, steps, train_cost, train_ler, val_cost, val_ler))


if __name__ == '__main__':
    train()









