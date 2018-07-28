import tensorflow as tf
import numpy as np


# input_size=6
# hidden_size_1=8
# hidden_size_2=3



class Autoencoder(object):
    """
      Autoencoder
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        # taking input_size as input and returning rmse error


        tf.reset_default_graph()
        x_data = tf.placeholder(name='Input_data', shape=[None, input_size], dtype=tf.float32)

        self.placeholder = {'input': x_data}

        # hidden layers for encoder and decoder part

        weights = {

            'encoder_1': tf.get_variable(name='encoder_1', shape=[input_size, hidden_size_1],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'encoder_2': tf.get_variable(name='encoder_2', shape=[hidden_size_1, hidden_size_2],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'decoder_1': tf.get_variable(name='decoder_2', shape=[hidden_size_2, hidden_size_1],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'decoder_2': tf.get_variable(name='decoder_3', shape=[hidden_size_1, input_size],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
        }

        bias = {

            'e_bias_1': tf.get_variable(name='e_encoder_1', shape=[hidden_size_1],
                                        dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'e_bias_2': tf.get_variable(name='e_encoder_2', shape=[hidden_size_2],
                                        dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'd_bias_1': tf.get_variable(name='e_decoder_2', shape=[hidden_size_1],
                                        dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)),
            'd_bias_2': tf.get_variable(name='e_decoder_3', shape=[input_size],
                                        dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
        }

        # encoder part
        def encoder_network(x):
            layer_first = tf.matmul(x, weights['encoder_1'])
            layer_first = tf.add(layer_first, bias['e_bias_1'])
            sigmoid_l = tf.nn.tanh(layer_first)

            layer_second = tf.matmul(sigmoid_l, weights['encoder_2'])
            layer_second = tf.add(layer_second, bias['e_bias_2'])
            return tf.nn.tanh(layer_second)

        # decoder part
        def decoder_network(y):
            decoder_first = tf.matmul(y, weights['decoder_1'])
            decoder_first = tf.add(decoder_first, bias['d_bias_1'])
            sigmoid_d = tf.nn.tanh(decoder_first)

            decoder_second = tf.matmul(sigmoid_d, weights['decoder_2'])
            decoder_second = tf.add(decoder_second, bias['d_bias_2'])
            return tf.nn.tanh(decoder_second)

        encoder_op = encoder_network(x_data)
        decoder_op = decoder_network(encoder_op)

        # Prediction
        y_pred = decoder_op

        # Targets (Labels) are the input data.
        y_true = x_data

        # Define batch mse
        batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1,name='output_batch')

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        self.output = {

                        'cost': cost,
                        'optimizer': optimizer,
                     }
        
        self.test = {
                       'batch_mse':batch_mse
                    }


# def train_model(model):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         c = sess.run(model.output, feed_dict={model.placeholder['input']: np.random.randint(0, 10, [12, 6])})
#         print(c)


# if __name__ == "__main__":
#     model = Autoencoder(6, 8, 3)

#     train_model(model)



