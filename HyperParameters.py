import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr
import tensorflow_probability as tfp

dis_opt = kr.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
cla_opt = kr.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
gen_opt = kr.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99,
                              use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)

cnt_dim = 256
ctg_dim = 16
ctg_w = 1.0
adv_reg_w = 1.0

is_mnist = False

is_cgpgan = True
is_gumbel = False

if is_gumbel:
    temperature = 0.1
    gumbel = tfp.distributions.Gumbel(loc=0.0, scale=1.0)
    ctg_prob_opt = kr.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
    ctg_log_prob = tf.Variable(tf.fill([ctg_dim], 1.0 / ctg_dim))
else:
    ctg_prob = tf.Variable(tf.fill([ctg_dim], 1.0 / ctg_dim))
if is_cgpgan:
    decay_rate = 0.999
    ctg_reg_w = 0.1

ctg_update_start_epoch = 30

batch_size = 16
step_per_epoch = 2000
epochs = 100

eval_model = True
epoch_per_eval = 10


def cnt_latent_dist(batch_size):
    return tf.random.normal([batch_size, cnt_dim])


def ctg_latent_dist(batch_size):
    if is_gumbel:
        return tf.nn.softmax((tf.math.log(tf.nn.softmax(ctg_log_prob[tf.newaxis])) + gumbel.sample([batch_size, ctg_dim])) / temperature, axis=-1)
    else:
        return tf.one_hot(tf.random.categorical(logits=[tf.math.log(ctg_prob + 1e-8)], num_samples=batch_size)[0], depth=ctg_dim)


def calc_ctg_ent():
    if is_gumbel:
        return tf.reduce_sum(-tf.nn.softmax(ctg_log_prob) * tf.math.log(tf.nn.softmax(ctg_log_prob) + 1e-8))
    else:
        return tf.reduce_sum(-ctg_prob * tf.math.log(ctg_prob + 1e-8))
