#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import tensorflow as tf


class SegmentationModel:
    def __init__(self, batch_size=24, number_of_class=2, kernel_size=5, depth=8, learning_rate=0.001):
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.number_of_class = number_of_class
        self.num_channels = 3
        self.graph = None
        self.tf_train_dataset = None
        self.tf_train_labels = None
        self.tf_train_weights = None
        self.tf_train_mode = None
        self.logits_train = None
        self.output_map = None
        self.optimizer = None
        self.loss = None
        self.saver = None
        self.build_graph()

    def build_graph(self):
        self.graph = tf.Graph()
        tf.reset_default_graph()

        with self.graph.as_default():
            tf_train_dataset = tf.placeholder(tf.float32, shape=(None, None, None, self.num_channels))
            tf_train_labels = tf.placeholder(tf.int32, shape=(None, None, None))
            tf_train_weights = tf.placeholder(tf.int32, shape=(None, None, None))
            tf_train_mode = tf.placeholder(tf.bool)
            
            logits_train = self.forward(tf_train_dataset, tf_train_mode)
            self.output_map = tf.nn.softmax(logits_train, name="output_map")
            loss = tf.losses.sparse_softmax_cross_entropy(tf_train_labels, logits_train, weights=tf_train_weights)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
            self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            self.tf_train_dataset = tf_train_dataset
            self.tf_train_labels = tf_train_labels
            self.tf_train_weights = tf_train_weights
            self.logits_train = logits_train
            self.optimizer = optimizer
            self.loss = loss
            self.tf_train_mode = tf_train_mode

    def convolution_step(self, data, depth, size=3, train=True):
        hidden = tf.layers.conv2d(data, depth, self.kernel_size, (1, 1), padding="SAME")
        hidden = tf.layers.batch_normalization(hidden, center=True, scale=False, training=train)
        return tf.nn.relu(hidden)

    def forward(self, data, train):
        hidden = data
        with tf.variable_scope('convolution_net', reuse=tf.AUTO_REUSE):
            for i in range(5):
                hidden = self.convolution_step(hidden, self.depth, train=train)
            return tf.layers.conv2d(hidden, self.number_of_class, 1, (1, 1), padding="SAME", name="final")
