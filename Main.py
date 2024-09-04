import Models
import HyperParameters as hp
import time
import Train
import tensorflow as tf
import numpy as np
import Evaluate


def main():
    np.set_printoptions(suppress=True)
    dis = Models.Encoder()
    cla = Models.Encoder()
    gen = Models.Generator()

    for epoch in range(hp.epochs):
        start = time.time()
        print('\nepoch', epoch)
        print('training...')
        train_results = Train.train(dis, cla, gen, epoch)
        print('time:', time.time() - start, '\n')

        gen_w = [tf.constant(w) for w in gen.trainable_variables]
        hp.gen_opt.finalize_variable_values(gen.trainable_variables)

        print('saving...')
        Evaluate.save_samples(gen, epoch)
        print('saved\n')

        if hp.eval_model and (epoch + 1) % hp.epoch_per_eval == 0:
            start = time.time()
            print('evaluating...')
            eval_results = Evaluate.eval(gen)
            Evaluate.save_figures(train_results, eval_results)
            print('time:', time.time() - start, '\n')

        for g_w, w in zip(gen.trainable_variables, gen_w):
            g_w.assign(w)


main()
