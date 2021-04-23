"""
TSGAN
Version: 2.0.0
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Note:
The timeseries GAN is based on a paper written in tensorflow 1 so it uses a lot of tensorflows
2.0's deprecated methods. Since it doesn't use keras (which is standard as of tf2.0) it might
be better to further optimize this for 2.0 (using keras and the object model yknow)
"""
import tensorflow as tf
import numpy as np
from tf_slim.layers import layers as _layers


class TSGAN():
    def __init__(self, hidden_dims, num_layers, iterations, batch_size, module_name):
        """
        This is the timeseries GAN class that provides an interface to create, train, save and
        inference from a model. Models are saved and accounted for in the `model
        """

        self.model_storage = "/directory/to/model/output/"

        self.hidden_dims = hidden_dims
        self.n_layers = num_layers
        self.iters = iterations
        self.batch_size = batch_size
        self.module_name = module_name
        self.input_shape = False
        self.session = False
        self.gamma = 1
        self.utils = GANUtils()

    def train(self, formatted_vector, output_filename):
        """
        This is the main training method of the function. It will train 3 tf variables according to the 
        parameters passed to the constructor. It will save the resulting weights and saved model states to 
        the directory `self.model_storage`/`output_filename`. Note** This function is not async! Do not call 
        in the main block! Instead delegate it to the model event daemon

        Parameters:
            - formatted_vector :: The vector that has already been formatted
            - output_filename :: A string of the resulting filename

        Returns:
            - True if the training was successful, otherwise it will raise an Exception()
        """

        # since we're using tensorflow 1's graph mode we gotta tell tensorflow 2 to chill out
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        self.input_shape = np.asarray(formatted_vector).shape
        time_vector, max_seq_len = self.utils.extract_time(formatted_vector)

        # normalize the data
        norm_vector, min_val, max_val = self.utils.scalar_min_max(
            formatted_vector)

        # instantiate the placeholder variables
        X = tf.compat.v1.placeholder(
            tf.float32, [None, max_seq_len, self.input_shape[2]], name="input_x")
        Z = tf.compat.v1.placeholder(
            tf.float32, [None, max_seq_len, self.input_shape[2]], name="input_z")
        T = tf.compat.v1.placeholder(tf.int32, [None], name="input_t")

        # Embedder & Recovery
        H = self._embedder(X, T)
        X_tilde = self._recovery(H, T)

        # Generator
        E_hat = self._generator(Z, T)
        H_hat = self._supervisor(E_hat, T)
        H_hat_supervise = self._supervisor(H, T)

        # Synthetic data
        X_hat = self._recovery(H_hat, T)

        # Discriminator
        Y_fake = self._discriminator(H_hat, T)
        Y_real = self._discriminator(H, T)
        Y_fake_e = self._discriminator(E_hat, T)

        e_vars = [v for v in tf.compat.v1.trainable_variables()
                  if v.name.startswith('embedder')]

        r_vars = [v for v in tf.compat.v1.trainable_variables()
                  if v.name.startswith('recovery')]
        g_vars = [v for v in tf.compat.v1.trainable_variables()
                  if v.name.startswith('generator')]
        s_vars = [v for v in tf.compat.v1.trainable_variables()
                  if v.name.startswith('supervisor')]
        d_vars = [v for v in tf.compat.v1.trainable_variables(
        ) if v.name.startswith('discriminator')]

        # discriminator loss
        D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.ones_like(Y_real), Y_real)
        D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.zeros_like(Y_fake), Y_fake)
        D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.zeros_like(Y_fake_e), Y_fake_e)
        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

        # generator loss
        # 1. Adversarial loss
        G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.ones_like(Y_fake), Y_fake)
        G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.ones_like(Y_fake_e), Y_fake_e)

        # 2. Supervised loss
        G_loss_S = tf.compat.v1.losses.mean_squared_error(
            H[:, 1:, :], H_hat_supervise[:, :-1, :])

        # 3. Two Moments
        G_loss_V1 = tf.reduce_mean(input_tensor=tf.abs(tf.sqrt(tf.nn.moments(
            x=X_hat, axes=[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(x=X, axes=[0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(input_tensor=tf.abs(
            (tf.nn.moments(x=X_hat, axes=[0])[0]) - (tf.nn.moments(x=X, axes=[0])[0])))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + self.gamma * G_loss_U_e + \
            100 * tf.sqrt(G_loss_S) + 100*G_loss_V

        # Embedder network loss
        E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
        E_loss0 = 10*tf.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1*G_loss_S

        # Embedder optimizers
        E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            E_loss0, var_list=e_vars + r_vars)

        E_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            E_loss, var_list=e_vars + r_vars)

        # Discriminator optimizer
        D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)

        # Gnerator optimizers
        G_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            G_loss, var_list=g_vars + s_vars)

        GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            G_loss_S, var_list=g_vars + s_vars)

        sess = tf.compat.v1.Session()

        sess.run(tf.compat.v1.global_variables_initializer())

        # first we train the embedding network
        for it in range(self.iters):

            # generate the training batch
            x_batch, t_batch = self.utils.batch_generator(
                formatted_vector, time_vector, self.batch_size)
            _, step_e_loss = sess.run(
                [E0_solver, E_loss_T0], feed_dict={X: x_batch, T: t_batch})

            if it % 1000 == 0:
                print('step: ' + str(it) + '/' + str(self.iters) +
                      ', e_loss: ' + str(np.round(np.sqrt(step_e_loss), 4)))

        print("Finished embedding network training\nStarting supervisor loss...")

        for itt in range(self.iters):
            # Set mini-batch
            X_mb, T_mb = self.utils.batch_generator(
                formatted_vector, time_vector, self.batch_size)
            # Random vector generation
            Z_mb = self.utils.rand_generator(
                self.batch_size, self.input_shape[2], T_mb, max_seq_len)
            # Train generator
            _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={
                                        Z: Z_mb, X: X_mb, T: T_mb})
            # Checkpoint
            if itt % 1000 == 0:
                print('step: ' + str(itt) + '/' + str(self.iters) +
                      ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s), 4)))

        print("Finished supervisor loss network training\nStarting joint loss...")
        for itt in range(self.iters):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                # Set mini-batch
                X_mb, T_mb = self.utils.batch_generator(
                    formatted_vector, time_vector, self.batch_size)
                # Random vector generation
                Z_mb = self.utils.rand_generator(
                    self.batch_size, self.input_shape[2], T_mb, max_seq_len)
                # Train generator
                _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
                    [G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
                # Train embedder
                _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={
                    Z: Z_mb, X: X_mb, T: T_mb})

            # Discriminator training
            # Set mini-batch
            X_mb, T_mb = self.utils.batch_generator(
                formatted_vector, time_vector, self.batch_size)
            # Random vector generation
            Z_mb = self.utils.rand_generator(
                self.batch_size, self.input_shape[2], T_mb, max_seq_len)
            # Check discriminator loss before updating
            check_d_loss = sess.run(
                D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
            # Train discriminator (only when the discriminator does not work well)
            if (check_d_loss > 0.15):
                _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={
                    X: X_mb, T: T_mb, Z: Z_mb})

            # Print multiple checkpoints
            if itt % 1000 == 0:
                print('step: ' + str(itt) + '/' + str(self.iters) +
                      ', d_loss: ' + str(np.round(step_d_loss, 4)) +
                      ', g_loss_u: ' + str(np.round(step_g_loss_u, 4)) +
                      ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) +
                      ', g_loss_v: ' + str(np.round(step_g_loss_v, 4)) +
                      ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0), 4)))
        print('Finish Joint Training')

        # save the session using the saver
        checkpoint_path = "{}{}".format(
            self.model_storage, output_filename)
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, checkpoint_path)
        print("Saved to directory {}".format(save_path))

        # generate some random data and return it for testing
        Z_mb = self.utils.rand_generator(
            self.input_shape[0], self.input_shape[2], time_vector, max_seq_len)
        generated_data = sess.run(
            X_hat, feed_dict={Z: Z_mb, X: formatted_vector, T: time_vector})

        gen = []
        for i in range(self.input_shape[0]):
            temp = generated_data[i, : time_vector[i], :]
            gen.append(temp)

        # renormalize the data
        gen = gen * max_val
        gen = gen + min_val

        return gen

    def _embedder(self, input_timeseries, input_time):
        """
        Embedding network that acts as an inbetween between the original feature space and the
        latent space.

        Arguments:
            - input_timeseries :: The input time-series data (shape must match with generator/discriminator)
            - input_time :: The input time information (must match the input timeseries)

        Returns:
            - embedding :: The generated embeddings
        """
        with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
            # testing here

            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [self.utils.rnn_cell(self.module_name, self.hidden_dims) for _ in range(self.n_layers)])

            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, input_timeseries, dtype=tf.float32, sequence_length=input_time)

            embedding = _layers.fully_connected(
                e_outputs, self.hidden_dims, activation_fn=tf.nn.sigmoid)
        return embedding

    def _recovery(self, H, T):
        """
        Recovery network stolen from Jinsung Yoon's original github repo idk wtf it does
        """
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [self.utils.rnn_cell(self.module_name, self.hidden_dims) for _ in range(self.n_layers)])
            r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(
                r_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = _layers.fully_connected(
                r_outputs, self.input_shape[2], activation_fn=tf.nn.sigmoid)
        return X_tilde

    def _generator(self, input_tensor, time_tensor):
        """
        Main generator function.

        Parameters:
            - input_tensor :: A random input tensor to generate the synthetic data from
            - time_tensor :: The stochstic information relating to the input
        """

        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [self.utils.rnn_cell(self.module_name, self.hidden_dims)
                 for _ in range(self.n_layers)]
            )
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                net, input_tensor, dtype=tf.float32, sequence_length=time_tensor)
            E = _layers.fully_connected(
                e_outputs, self.hidden_dims, activation_fn=tf.nn.sigmoid)
        return E

    def _supervisor(self, latent_tensor, time_tensor):
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [self.utils.rnn_cell(self.module_name, self.hidden_dims) for _ in range(self.n_layers-1)])
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, latent_tensor, dtype=tf.float32, sequence_length=time_tensor)
            S = _layers.fully_connected(
                e_outputs, self.hidden_dims, activation_fn=tf.nn.sigmoid)
        return S

    def _discriminator(self, latent_tensor, time_tensor):
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [self.utils.rnn_cell(self.module_name, self.hidden_dims) for _ in range(self.n_layers)])
            d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(
                d_cell, latent_tensor, dtype=tf.float32, sequence_length=time_tensor)
            Y_hat = _layers.fully_connected(
                d_outputs, 1, activation_fn=None)
        return Y_hat


class GANUtils():
    """
    A utility library used by the GAN
    """

    def rnn_cell(self, module_name, hidden_dim):
        # TODO Since this model defaults to using CUDNNGRU (which is only)
        # available on CUDA, we need a way to switch between GRUCell and
        # CUDNNGRU

        if module_name not in ['gru', 'lstm', 'lstmln']:
            return False

        if module_name == 'gru':

            # testing
            rnn_cell = tf.compat.v1.nn.rnn_cell.GRUCell(
                num_units=hidden_dim, activation=tf.nn.tanh)
        else:
            rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                num_units=hidden_dim, activation=tf.nn.tanh)

        return rnn_cell

    def min_max(self, data):
        """
        Minmax normalizes the input vector
        """
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        norm_data = numerator / (denominator + 1e-7)
        return norm_data

    def scalar_min_max(self, data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    def batch_generator(self, data, time, batch_size):
        """
        Generates a batch for training
        """
        no = len(data)
        idx = np.random.permutation(no)
        train_idx = idx[:batch_size]

        X_mb = list(data[i] for i in train_idx)
        T_mb = list(time[i] for i in train_idx)

        return X_mb, T_mb

    def format_stock_data(self, stock_object, sequence_length=24):
        """
        Formats and loads the stock data

        Parameters:
            - stock_object :: A dictionary of stocks with each key being the ticker and each value
                              - being an array of stock object with the keys: 
                                            [open, high, low, close, volume]
            - sequence_length :: The length of which each batch of data will be (integer)

        Returns:
            - data :: The preprocessed data. A dictionary with each key being the ticker and each val 
                      being that tickers list of data
        """
        data = {}
        for ticker in stock_object:
            # the shape of this vector is (num_rows, 5) with 5 being the
            # open,high,low,close and volume values for that day
            vector = np.array([[x['open'], x['high'], x['low'],
                                x['close'], x['volume']] for x in stock_object[ticker]])

            vector = vector[::-1]
            vector = self.min_max(vector)

            preproc = []
            for i in range(0, len(vector) - sequence_length):
                v = vector[i:i + sequence_length]
                preproc.append(v)

            rand = np.random.permutation(len(preproc))
            data = []
            for i in range(len(preproc)):
                data.append(preproc[rand[i]])

            return data

    def extract_time(self, data):
        """
        Extracts the sequence length and time from the data.
        """
        time = list()
        max_seq_len = 0
        for i in range(len(data)):
            max_seq_len = max(max_seq_len, len(data[i][:, 0]))
            time.append(len(data[i][:, 0]))

        return time, max_seq_len

    def rand_generator(self, batch_size, z_dim, T_mb, max_seq_len):
        Z_mb = list()
        for i in range(batch_size):
            temp = np.zeros([max_seq_len, z_dim])
            temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
            temp[:T_mb[i], :] = temp_Z
            Z_mb.append(temp_Z)
        return Z_mb
