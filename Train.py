import tensorflow as tf
import Dataset
import HyperParameters as hp


@tf.function
def _cgpgan_train_step(dis, cla, gen, update_ctg_prob):
    real_x = Dataset.data_dist(hp.batch_size)
    cnt_vecs = hp.cnt_latent_dist(hp.batch_size)
    ctg_vecs = hp.ctg_latent_dist(hp.batch_size)
    fake_x = gen([ctg_vecs, cnt_vecs])

    with tf.GradientTape(persistent=True) as dis_cla_tape:
        real_ctg_probs = tf.nn.softmax(cla(real_x))
        if not update_ctg_prob:
            real_ctg_probs = real_ctg_probs - tf.reduce_mean(real_ctg_probs, axis=0, keepdims=True) + 1.0 / hp.ctg_dim
        real_ctg_vecs = tf.one_hot(tf.argmax(real_ctg_probs, axis=-1), depth=hp.ctg_dim)
        real_adv_vals = tf.reduce_sum(dis(real_x) * real_ctg_vecs, axis=-1)

        with tf.GradientTape(persistent=True) as reg_tape:
            reg_tape.watch(fake_x)
            fake_ctg_logits = cla(fake_x)
            ctg_losses = tf.losses.categorical_crossentropy(ctg_vecs, fake_ctg_logits, from_logits=True)
            fake_adv_vals = tf.reduce_sum(dis(fake_x) * ctg_vecs, axis=-1)
            ctg_reg_scores = tf.square(1 - tf.reduce_sum(tf.nn.softmax(fake_ctg_logits) * ctg_vecs, axis=-1))

        if hp.is_mnist:
            adv_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(fake_adv_vals, fake_x)), axis=[1, 2, 3])
            ctg_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(ctg_reg_scores, fake_x)), axis=[1, 2, 3])
        else:
            adv_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(fake_adv_vals, fake_x)), axis=-1)
            ctg_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(ctg_reg_scores, fake_x)), axis=-1)

        dis_adv_losses = tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals)

        dis_losses = dis_adv_losses + hp.adv_reg_w * adv_reg_losses
        cla_losses = hp.ctg_w * ctg_losses + hp.ctg_reg_w * ctg_reg_losses

        dis_loss = tf.reduce_mean(dis_losses)
        cla_loss = tf.reduce_mean(cla_losses)

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_cla_tape)
    hp.cla_opt.minimize(cla_loss, cla.trainable_variables, tape=dis_cla_tape)

    acc = 1 - tf.math.count_nonzero(tf.argmax(fake_ctg_logits, axis=-1) - tf.argmax(ctg_vecs, axis=-1)) / hp.batch_size

    cnt_vecs = hp.cnt_latent_dist(hp.batch_size)
    ctg_vecs = hp.ctg_latent_dist(hp.batch_size)

    with tf.GradientTape() as gen_tape:
        fake_x = gen([ctg_vecs, cnt_vecs])
        fake_adv_vals = tf.reduce_sum(dis(fake_x) * ctg_vecs, axis=-1)
        gen_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vals))

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)
    hp.ctg_prob.assign(hp.ctg_prob * hp.decay_rate + tf.reduce_mean(real_ctg_probs, axis=0) * (1 - hp.decay_rate))

    results = {'real_adv_vals': real_adv_vals, 'fake_adv_vals': fake_adv_vals, 'ctg_losses': ctg_losses,
               'adv_reg_losses': adv_reg_losses, 'ctg_reg_losses': ctg_reg_losses, 'acc': [acc]}

    return results


@tf.function
def _infogan_train_step(dis, cla, gen, update_ctg_prob):
    real_x = Dataset.data_dist(hp.batch_size)
    cnt_vecs = hp.cnt_latent_dist(hp.batch_size)
    ctg_vecs = hp.ctg_latent_dist(hp.batch_size)
    fake_x = gen([ctg_vecs, cnt_vecs])

    with tf.GradientTape(persistent=True) as dis_tape:
        with tf.GradientTape() as reg_tape:
            reg_tape.watch(fake_x)
            fake_adv_vals = dis(fake_x)[:, 0]

        if hp.is_mnist:
            adv_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(fake_adv_vals, fake_x)), axis=[1, 2, 3])
        else:
            adv_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(fake_adv_vals, fake_x)), axis=-1)

        real_adv_vals = dis(real_x)[:, 0]
        fake_ctg_logits = cla(fake_x)

        ctg_losses = tf.losses.categorical_crossentropy(ctg_vecs, fake_ctg_logits, from_logits=True)
        dis_adv_losses = tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals)

        dis_losses = dis_adv_losses + hp.adv_reg_w * adv_reg_losses
        cla_losses = hp.ctg_w * ctg_losses

        dis_loss = tf.reduce_mean(dis_losses)
        cla_loss = tf.reduce_mean(cla_losses)

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_tape)
    hp.cla_opt.minimize(cla_loss, cla.trainable_variables, tape=dis_tape)

    cnt_vecs = hp.cnt_latent_dist(hp.batch_size)

    with tf.GradientTape(persistent=True) as gen_tape:
        ctg_vecs = hp.ctg_latent_dist(hp.batch_size)
        fake_x = gen([ctg_vecs, cnt_vecs])
        fake_ctg_logits = cla(fake_x)
        fake_adv_vals = dis(fake_x)[:, 0]

        ctg_losses = tf.losses.categorical_crossentropy(ctg_vecs, fake_ctg_logits, from_logits=True)

        gen_adv_losses = tf.nn.softplus(-fake_adv_vals)

        gen_losses = gen_adv_losses + hp.ctg_w * ctg_losses
        gen_loss = tf.reduce_mean(gen_losses)
        if update_ctg_prob:
            ctg_prob_loss = gen_loss
        else:
            ctg_prob_loss = gen_loss * 0.0

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)
    if hp.is_gumbel:
        hp.ctg_prob_opt.minimize(ctg_prob_loss, [hp.ctg_log_prob], tape=gen_tape)

    acc = 1 - tf.math.count_nonzero(tf.argmax(fake_ctg_logits, axis=-1) - tf.argmax(ctg_vecs, axis=-1)) / hp.batch_size

    results = {'real_adv_vals': real_adv_vals, 'fake_adv_vals': fake_adv_vals, 'ctg_losses': ctg_losses,
               'adv_reg_losses': adv_reg_losses, 'acc': [acc]}

    return results


def train(dis, cla, gen, epoch):
    results = {}
    for i in range(hp.step_per_epoch):
        if hp.is_cgpgan:
            batch_results = _cgpgan_train_step(dis, cla, gen, hp.ctg_update_start_epoch <= epoch)
        else:
            batch_results = _infogan_train_step(dis, cla, gen, hp.ctg_update_start_epoch <= epoch)

        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results
    results['ctg_ent'] = hp.calc_ctg_ent()

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    if hp.is_gumbel:
        ctg_prob = tf.nn.softmax(hp.ctg_log_prob)
    else:
        ctg_prob = hp.ctg_prob
    print('ctg_prob:', ctg_prob.numpy())

    return results
