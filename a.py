import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.random.normal([10000, 20000])
    b = tf.random.normal([20000, 200])
    print(a, b)


def run():
    with tf.device('/gpu:0'):
        c = tf.matmul(a, b)
        print((c))


if __name__ == '__main__':
    run()
