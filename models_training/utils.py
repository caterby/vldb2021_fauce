import numpy as np
import pandas as pd
#import ipdb


class DataLoader_RegressionToy():

    def __init__(self, args):

        assert args.dataset in ['five_joins_012345_5_12_filters_error_log10']

        if args.dataset == 'five_joins_012345_5_12_filters_error_log10':
            self.data = pd.read_csv('/home/bo/deep-ensembles-uncertainty/datasets/JOB_five_joins_training_dataset_400000_9_16filers_log15.csv')

        self.column_names = self.data.columns
        self.num_rows = (int)(len(self.data)/1)
        print(self.num_rows)
        self.num_columns = len(self.column_names)
        print(self.num_columns)
        self.num_features = self.num_columns - 1
        self.data_x = np.zeros([self.num_rows, self.num_features])
        self.data_y = np.zeros([self.num_rows, 1])

        for i in range(self.num_rows):
            for j in range(self.num_features):
                self.data_x[i][j] = self.data[self.column_names[j]][i]
            self.data_y[i,0] = self.data[self.column_names[-1]][i]

        print('The shape of the data_x is: ' + str(self.data_x.shape))
        print(self.data_y)

        self.min_val = self.data_y.min()
        self.max_val = self.data_y.max()
        print('self.min_val is: {}'.format(self.min_val))
        print('self.max_val is: {}'.format(self.max_val))

        self.data_y = (self.data_y - self.min_val)/(self.max_val - self.min_val)

        self.num_train_data = int(self.num_rows * (1 - args.test_ratio))

        # Standardize input features
        self.input_mean = np.mean(self.data_x, 0)
        self.input_std = np.std(self.data_x, 0)
        self.data_x_standardized = (self.data_x - self.input_mean)/self.input_std

        # Training data
        self.train_data_x = self.data_x_standardized[:self.num_train_data, :]
        self.train_data_y = self.data_y[:self.num_train_data, :]

        print('The shape of the train_data_x is: ' + str(self.train_data_x.shape))

        # Target mean and std
        self.target_mean = np.mean(self.train_data_y, 0)[0]
        self.target_std = np.std(self.train_data_y, 0)[0]

        # Testing data
        self.test_data_x = self.data_x_standardized[self.num_train_data:, :]
        self.test_data_y = self.data_y[self.num_train_data:, :]

        self.batch_size = args.batch_size
        self.num_test_data = args.num_test_data


    def next_batch(self):

        train_indices = np.random.choice(np.arange(len(self.train_data_x)), size=self.batch_size)
        train_x = np.zeros([self.batch_size, self.train_data_x.shape[1]])
        train_y = np.zeros([self.batch_size, 1])

        for i in range(train_indices.shape[0]):
            train_x[i, :] = self.train_data_x[train_indices[i], :]
            train_y[i, :] = self.train_data_y[train_indices[i], :]

        
        return train_x, train_y


    def get_data(self):

        return self.data_x_standardized, self.data_y


    def get_min_max(self):
        return self.min_val, self.max_val


    def get_test_data(self):

        test_indices = [i for i in range(self.num_test_data)]

        test_x = np.zeros([self.num_test_data, self.train_data_x.shape[1]])
        test_y = np.zeros([self.num_test_data, 1])

        for i in range(len(test_indices)):
            test_x[i, :] = self.test_data_x[test_indices[i], :]
            test_y[i, :] = self.test_data_y[test_indices[i], :]

        return test_x, test_y


    def get_num_features(self):
        
        return self.num_features



class DataLoader_RegressionToy_withKink():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-1, 1, num=1000, dtype=np.float32), -1)

        self.ys = np.zeros(shape=self.xs.shape)
        for i, t in enumerate(self.xs):
            if t > 0.25 or t < -0.25:
                self.ys[i] = 10*(t)**3 + np.random.normal(scale=.1)
            else:
                self.ys[i] = 30*np.sin(t) + np.random.normal(scale=.1)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):
        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-1.5, 1.5, num=1000, dtype=np.float32), -1)

        test_ys = np.zeros(shape=test_xs.shape)
        for i, t in enumerate(test_xs):
            if t > 0.25 or t < -0.25:
                test_ys[i] = 10*(t)**3 + np.random.normal(scale=.1)
            else:
                test_ys[i] = 30*np.sin(t) + np.random.normal(scale=.1)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_sinusoidal():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-8, 8, num=1000, dtype=np.float32), -1)

        self.ys = 5*(np.sin(self.xs)) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-16, 16, num=2000, dtype=np.float32), -1)

        test_ys = 5*(np.sin(test_xs)) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_sinusoidal_break():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-8, 8, num=1000, dtype=np.float32), -1)

        self.xs = np.expand_dims(np.delete(self.xs, np.arange(300, 700)), -1)

        self.ys = 5*(np.sin(self.xs)) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-8, 8, num=2000, dtype=np.float32), -1)

        test_ys = 5*(np.sin(test_xs)) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_break():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-4, 4, num=100, dtype=np.float32), -1)

        self.xs = np.expand_dims(np.delete(self.xs, np.arange(40, 90)), -1)

        self.ys = 5*(self.xs**3) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-4, 4, num=200, dtype=np.float32), -1)

        test_ys = 5*(test_xs**3) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys
