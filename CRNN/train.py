import cv2
import tensorflow as tf
from CRNN.model import Model
from CRNN.util import Util, sparseTensor_to_seq, Test
import os

# 使用第一块GPU进行训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


GPU_MEMORY_FRACTION = 0.85 # gpu内存
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
BATCH_SIZE = 5
EPOCH_NUM = 10000
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

    # 划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    # dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    # 计算两个序列之间的Levenshtein 距离
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
    # 配置gpu训练参数
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        global_step = tf.Variable(0, trainable=False)
        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for i in range(EPOCH_NUM):
            train_cost = train_ler = 0
            train_inputs, train_targets, train_seq_len = u.get_next_batch(BATCH_SIZE)
            val_feed = {inputs: train_inputs, labels: train_targets, seq_len: train_seq_len}
            val_cost, val_ler, steps, _ = sess.run([loss, acc, global_step, optimizer], feed_dict=val_feed)
            print("Epoch.......", i)
            if i % 10 == 0:
                u = Util()
                test_inputs, test_targets, test_seq_len = u.get_next_batch(BATCH_SIZE)
                test_feed = {inputs: test_inputs,
                             labels: test_targets,
                             seq_len: test_seq_len}
                dd, log_probs, accuracy = sess.run([decoded[0], log_prob, acc], test_feed)
                report_accuracy(dd, test_targets)
                saver.save(sess, "./models/crnn.ckpt", global_step=steps)
            log = "Epoch {}/{}, val_cost = {:.3f}, edit_distance = {:.3f}"
            print(log.format(i + 1, EPOCH_NUM, val_cost, val_ler))
        saver.save(sess, "./models/crnn.ckpt")
        print("训练完成！！")


# 计算准确率
def report_accuracy(decoded_list, test_targets):
    original_list = sparseTensor_to_seq(test_targets)
    detected_list = sparseTensor_to_seq(decoded_list)
    true_numer = 0
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list), " test and detect length desn't match")
        return \
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit: true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))


# 评估函数
def evaluate(img_dir, batch_size):
    with tf.Session() as sess:
        t = Test(256, 64, batch_size, "D:\\work\\tianchi\\train_set\\")
        u = Util()
        # image输入
        inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        seq_len = tf.placeholder(tf.int32, [None])
        # 获取模型
        model = Model(hidden_nums=256, seq_length=seq_len, num_classes=u.maxlength)
        logits, raw_pred = model.build_model(inputs)
        # 划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        # 恢复模型
        saver = tf.train.Saver()
        saver.restore(sess, './models/./crnn.ckpt-0')
        epochs = t.nums // batch_size + 1
        for epoch in range(epochs):
            names, imgs, seq_length = t.get_test_batch()
            dd, log_probs = sess.run([decoded[0], log_prob], feed_dict={inputs: imgs, seq_len: seq_length})
            seq = sparseTensor_to_seq(dd)
            save_result(names, seq)
            print(seq)


def save_result(names,seqs):
    labelfile = open('result.txt', 'a')
    for i, name in enumerate(names):
        value = name + ' ' + seqs[i] + '\n'
        value.encode("utf-8")
        labelfile.write(value)
    labelfile.close()
    print("成功保存一个batch！")


if __name__ == '__main__':
    train()
    # evaluate("D:\\work\\tianchi\\train_set", 5)
    # n = ["1", "3", "5"]
    # v = ['a', 'b', 'c']
    # save_result(n,v)








