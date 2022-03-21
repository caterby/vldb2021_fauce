import tensorflow as tf
#tf.disable_v2_behavior()
#import tensorflow.compat.v1 as tf tf.disable_v2_behavior()

#import tensorflow.compat.v1 as tf
import numpy as np

#tf.disable_v2_behavior()

class MLPGaussianRegressor():

    def __init__(self, args, sizes, model_scope):

        self.input_data = tf.placeholder(tf.float32, [None, sizes[0]])
        self.target_data = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')

        with tf.variable_scope(model_scope+'target_stats'):
            self.output_mean = tf.Variable(0., trainable=False, dtype=tf.float32)
            self.output_std = tf.Variable(0.1, trainable=False, dtype=tf.float32)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random.normal([sizes[i-1], sizes[i]], stddev=0.1), name='weights_'+str(i-1)))
                self.biases.append(tf.Variable(tf.random.normal([sizes[i]], stddev=0.1), name='biases_'+str(i-1)))
                #initializer = tf.initializers.GlorotUniform()
                #initializer = tf.contrib.layers.xavier_initializer()
                #shape_w = (sizes[i-1], sizes[i])
                #shape_b = (1, sizes[i])
                #var_w = tf.Variable(initializer(shape = shape_w), name='weights_'+str(i-1))
                #var_b = tf.Variable(initializer(shape = shape_b), name='biases_'+str(i-1))
                #self.weights.append(var_w)
                #self.biases.append(var_b)

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = tf.nn.relu(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.raw_var = tf.split(self.output, [1,1], axis=1)

        # Output transform
        self.mean = self.mean * self.output_std + self.output_mean
        self.var = (tf.math.log(1 + tf.exp(self.raw_var)) + 1e-6) * (self.output_std**2)

        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.subtract(y, mean_values)
            print('The var_values is: {}'.format(var_values))
            return 0.5*tf.reduce_mean(input_tensor=tf.math.log(var_values+args.beta)) + 0.5*tf.reduce_mean(input_tensor=tf.div(tf.square(y_diff), var_values)) + 0.5*tf.math.log(2*np.pi)

        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        self.nll_gradients = tf.gradients(ys=args.alpha * self.nll, xs=self.input_data)[0]
        print('self.nll_gradients is: {}'.format(self.nll_gradients))

        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            x_at = tf.nn.relu(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))

        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

        mean_at, raw_var_at = tf.split(output_at, [1, 1], axis=1)

        # Output transform
        mean_at = mean_at * self.output_std + self.output_mean
        var_at = (tf.math.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

        self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)

        tvars = tf.compat.v1.trainable_variables()

        for v in tvars:
            print(v.name)
            print(v.get_shape())

        self.gradients = tf.gradients(ys=args.alpha * self.nll + (1 - args.alpha) * self.nll_at, xs=tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))


class MLPDropoutGaussianRegressor():

    def __init__(self, args, sizes, model_scope):

        self.input_data = tf.compat.v1.placeholder(tf.float32, [None, sizes[0]])
        self.target_data = tf.compat.v1.placeholder(tf.float32, [None, sizes[0]])

        with tf.compat.v1.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')

        with tf.compat.v1.variable_scope(model_scope+'target_stats'):
            self.output_mean = tf.Variable(0., trainable=False, dtype=tf.float32)
            self.output_std = tf.Variable(0.1, trainable=False, dtype=tf.float32)

        self.dr = tf.compat.v1.placeholder(tf.float32)

        self.weights = []
        self.biases = []

        with tf.compat.v1.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random.normal([sizes[i-1], sizes[i]], stddev=0.1), name='weights_'+str(i-1)))
                self.biases.append(tf.Variable(tf.random.normal([sizes[i]], stddev=0.1), name='biases_'+str(i-1)))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = tf.nn.relu(tf.compat.v1.nn.xw_plus_b(x, self.weights[i], self.biases[i]))
            x = tf.nn.dropout(x, 1 - (self.dr), noise_shape=[1, sizes[i+1]], name='dropout_layer'+str(i))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.raw_var = tf.split(1, 2, self.output)

        # Output transform
        self.mean = self.mean * self.output_std + self.output_mean
        self.var = (tf.math.log(1 + tf.exp(self.raw_var)) + 1e-6) * (self.output_std**2)

        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.subtract(y, mean_values)
            return 0.5*tf.reduce_mean(input_tensor=tf.math.log(var_values)) + 0.5*tf.reduce_mean(input_tensor=tf.compat.v1.div(tf.square(y_diff), var_values)) + 0.5*tf.math.log(2*np.pi)

        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        self.nll_gradients = tf.gradients(ys=args.alpha * self.nll, xs=self.input_data)[0]
        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            x_at = tf.nn.relu(tf.compat.v1.nn.xw_plus_b(x_at, self.weights[i], self.biases[i]))
            # We need to apply the same dropout mask as before
            # so that we maintain the same model and not change the network
            graph = tf.compat.v1.get_default_graph()
            mask = graph.get_tensor_by_name('dropout_layer'+str(i)+'/Floor:0')
            x_at = tf.mul(x_at, mask)

        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

        mean_at, raw_var_at = tf.split(1, 2, output_at)

        # Output transform
        mean_at = mean_at * self.output_std + self.output_mean
        var_at = (tf.math.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

        self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)

        tvars = tf.compat.v1.trainable_variables()

        for v in tvars:
            print(v.name)
            print(v.get_shape())

        self.gradients = tf.gradients(ys=args.alpha * self.nll + (1 - args.alpha) * self.nll_at, xs=tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
