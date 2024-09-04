import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp

if hp.is_mnist:
    dataset = kr.datasets.mnist.load_data()
    dataset = tf.cast(tf.concat([dataset[0][0], dataset[1][0]], axis=0), 'float32')[:, :, :, tf.newaxis] / 127.5 - 1

    def data_dist(batch_size):
        indexes = tf.random.uniform(shape=[batch_size], maxval=dataset.shape[0], dtype='int64')
        return tf.gather(dataset, indexes)


else:
    cluster_coords = [(-1.0, 1.0), (0.0, -2.0), (1.0, -1.0), (2.0, 1.)]
    cluster_logits = tf.math.log([0.1, 0.2, 0.3, 0.4])
    cluster_rad = 0.3

    def data_dist(batch_size):
        indexes = tf.random.categorical([cluster_logits], batch_size)[0]
        coords = tf.gather(cluster_coords, indexes)
        coords += tf.random.normal(stddev=cluster_rad, shape=coords.shape)

        return coords
