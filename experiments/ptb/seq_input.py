import tensorflow as tf
import numpy as np
def ptb_raw_data(data, val_size = 0.1, test_size = 0.1):
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))
    print(nval, ntest)
    train_data, valid_data, test_data = data[:nval,:], data[nval:ntest,:], data[ntest:,:]
    return train_data, valid_data, test_data


def ptb_producer(raw_data, batch_size, num_steps, name):
    """The mini-batch generator"""
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        (data_len,data_dim) = np.shape(raw_data)
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)

        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len,:],
                          [batch_size, batch_len, -1])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps,0], [batch_size, num_steps, data_dim])
        y = tf.slice(data, [0, i * num_steps + 1,0], [batch_size, num_steps, data_dim])
        return x, y
    
class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = ptb_producer(
            data, batch_size, num_steps, name=name)
