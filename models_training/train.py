import numpy as np
import argparse
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import os
import math
#import matplotlib.pyplot as plt

#import ipdb
from pandas import DataFrame
from model import MLPGaussianRegressor
from model import MLPDropoutGaussianRegressor

from utils import DataLoader_RegressionToy
from utils import DataLoader_RegressionToy_withKink
from utils import DataLoader_RegressionToy_sinusoidal
from utils import DataLoader_RegressionToy_sinusoidal_break
from utils import DataLoader_RegressionToy_break

#tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def main():

    parser = argparse.ArgumentParser()
    # Training Dataset
    parser.add_argument('--dataset', type=str, default='five_joins_012345_5_12_filters_error_log10',
                        help='Name of the input training dataset')
    # Ensemble size
    parser.add_argument('--ensemble_size', type=int, default=10,
                        help='Size of the ensemble')
    # Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=30000,
                        help='Maximum number of iterations')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of batch')
    # Epsilon for adversarial input perturbation
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Epsilon for adversarial input perturbation')
    # Alpha for trade-off between likelihood score and adversarial score
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Trade off parameter for likelihood score and adversarial score')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimization')
    # Gradient clipping value
    parser.add_argument('--grad_clip', type=float, default=100.,
                        help='clip gradients at this value')
    # Learning rate decay
    parser.add_argument('--decay_rate', type=float, default=0.98,
                        help='Decay rate for learning rate')
    # Dropout rate (keep prob)
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probability for dropout')
    # Testing ratio (keep prob)
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Keep probability for test ratio')
    # Number of data to be tested after training
    parser.add_argument('--num_test_data', type=int, default=6000,
                        help='The number of data to be tested')
    # Beta for avoiding the NaN for the loss
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='This small number added into the loss for avoiding Nan')
    args = parser.parse_args()
    train_ensemble(args)
    # train_dropout(args)


def ensemble_mean_var(ensemble, xs, sess):
    en_mean = 0
    en_var = 0
    total = 0

    for model in ensemble:
        feed = {model.input_data: xs}
        mean, var = sess.run([model.mean, model.var], feed)
        
        is_mean_nan = np.isnan(mean).any()
        is_var_nan = np.isnan(var).any()

        if is_mean_nan == False and is_var_nan == False:
            en_mean += mean
            en_var += var + mean**2
            total += 1
        else:
            continue
        

    en_mean /= total
    en_var /= total
    en_var -= en_mean**2
    print('The total number of none nan value is: {}'.format(total))
    return en_mean, en_var


def dropout_mean_var(model, xs, sess, args):
    en_mean = 0
    en_var = 0
    total = 0

    for i in range(args.ensemble_size):
        # NOTE using dropout at test time as well
        feed = {model.input_data: xs, model.dr: args.keep_prob}
        mean, var = sess.run([model.mean, model.var], feed)
        
        is_mean_nan = np.isnan(mean).any()
        is_var_nan = np.isnan(var).any()

        if is_mean_nan == False and is_var_nan == False:
            en_mean += mean
            en_var += var + mean**2
            total += 1
        else:
            continue

    en_mean /= total
    en_var /= total
    en_var -= en_mean**2
    return en_mean, en_var


def train_ensemble(args):
    
    # Input data
    dataLoader = DataLoader_RegressionToy(args)
    num_features = dataLoader.get_num_features()

    # Layer sizes
    sizes = [num_features, 128, 256, 512, 512, 2]

    ensemble = [MLPGaussianRegressor(args, sizes, 'model'+str(i)) for i in range(args.ensemble_size)]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for model in ensemble:
            sess.run(tf.assign(model.output_mean, dataLoader.target_mean))
            sess.run(tf.assign(model.output_std, dataLoader.target_std))

        for itr in range(args.max_iter):

            for model in ensemble:

                x, y = dataLoader.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll, m, v = sess.run([model.train_op, model.nll, model.mean, model.var], feed)

                if itr % 300 == 0:
                    sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** (itr/300))))
                    print('itr: {}, nll: {}'.format(itr, nll))

        test_ensemble(ensemble, sess, dataLoader)


def test_ensemble(ensemble, sess, dataLoader):
    test_xs, test_ys = dataLoader.get_test_data()
    min_val, max_val = dataLoader.get_min_max()
    print('min_val is : {}'.format(min_val))
    print('max_val is : {}'.format(max_val))

    mean, var = ensemble_mean_var(ensemble, test_xs, sess)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std
    
    test_xs_scaled = dataLoader.input_mean + dataLoader.input_std*test_xs

    estimate_mean1 = [i * (max_val - min_val) + min_val for i in mean]
    estimate_mean2 = [np.round(np.power(15, i)) for i in estimate_mean1]

    test_ys1 = [i * (max_val - min_val) + min_val for i in test_ys]
    test_ys2 = [np.round(np.power(15, i)) for i in test_ys1]
    
    final_error = []

    for i in range(2000):
        temp_estimate_val = estimate_mean2[i]
        temp_real_val = test_ys2[i]

        if temp_estimate_val == 0 and temp_real_val != 0:
            error = temp_real_val
        elif temp_estimate_val != 0 and temp_real_val == 0:
            error = temp_estimate_val
        elif temp_estimate_val == 0 and temp_real_val == 0:
            error = 1
        elif temp_estimate_val != 0 and temp_real_val != 0:
            error = max(temp_estimate_val/temp_real_val, temp_real_val/temp_estimate_val)

        final_error.append(error)


    final_test_error_write = np.array(final_error)
    final_test_error_cols_num = final_test_error_write.shape[1]
    final_test_error_cols_name = ['error{}'.format(i) for i in range(final_test_error_cols_num)]
    final_test_error_dataToCSV = DataFrame(final_test_error_write, columns = final_test_error_cols_name)
    final_test_error_dataToCSV.to_csv('/home/bo/deep-ensembles-uncertainty/results/version2_five_joins_400000_9_16_filters_error_log15_itr20k_train400k_ens10_128_256_512_512_lre-4_xavier.csv', escapechar=None)
    print('Final testing error write is done!')
    
    #plt.plot(test_xs_scaled, test_ys, 'b-')
    #plt.plot(test_xs_scaled, mean, 'r-')

    #plt.fill_between(test_xs_scaled[:, 0], lower[:, 0], upper[:, 0], color='yellow', alpha=0.5)
    #plt.show()


def train_dropout(args):
    
    # Input data
    dataLoader = DataLoader_RegressionToy(args)
    num_features = dataLoader.get_num_features()

    # Layer sizes
    sizes = [num_features, 256, 512, 512, 2]

    model = MLPDropoutGaussianRegressor(args, sizes, 'dropout_model')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.assign(model.output_mean, dataLoader.target_mean))
        sess.run(tf.compat.v1.assign(model.output_std, dataLoader.target_std))
        for itr in range(args.max_iter):

            x, y = dataLoader.next_batch()

            feed = {model.input_data: x, model.target_data: y, model.dr: args.keep_prob}
            _, nll = sess.run([model.train_op, model.nll], feed)

            if itr % 300 == 0:
                sess.run(tf.compat.v1.assign(model.lr, args.learning_rate * (args.decay_rate ** (itr/300))))
                print('itr: {}, nll: {}'.format(itr, nll))

        test_dropout(model, sess, dataLoader, args)


def test_dropout(model, sess, dataLoader, args):
    test_xs, test_ys = dataLoader.get_test_data()
    mean, var = dropout_mean_var(model, test_xs, sess, args)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std

    test_xs_scaled = dataLoader.input_mean + dataLoader.input_std*test_xs

    #plt.plot(test_xs_scaled, test_ys, 'b-')
    #plt.plot(test_xs_scaled, mean, 'r-')

    #plt.fill_between(test_xs_scaled[:, 0], lower[:, 0], upper[:, 0], color='yellow', alpha=0.5)
    #plt.show()

if __name__ == '__main__':
    main()
