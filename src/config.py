from tensorflow.python.client import device_lib

class Config(object):
    def __init__(self):
        self.train_steps = 50000000
        self.batch_size = 64
        self.history_len = 4
        self.frame_skip = 4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.max_steps = 10000
        self.epsilon_decay_episodes = 1000000
        self.train_freq = 8
        self.update_freq = 10000
        self.train_start = 20000
        self.dir_save = "saved_session/"
        self.restore = False
        self.epsilon_decay = float((self.epsilon_start - self.epsilon_end)) / float(self.epsilon_decay_episodes)
        self.random_start = 10
        self.test_step = 5000
        self.network_type = "drqn"

        self.gamma = 0.99
        self.learning_rate_minimum = 0.00025
        self.lr_method = "rmsprop"
        self.learning_rate = 0.00025
        self.lr_decay = 0.97
        self.keep_prob = 0.8

        self.num_lstm_layers = 1
        self.lstm_size = 512
        self.min_history = 4
        self.states_to_update = 4


        if self.get_available_gpus():
            self.cnn_format = "NCHW"
        else:
            self.cnn_format = "NHWC"

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']