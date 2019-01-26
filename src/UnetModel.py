from SegmentationModel import SegmentationModel
import tensorflow as tf


class UnetModel(SegmentationModel):
    def __init__(self, batch_size=24, number_of_class=2, kernel_size=5, depth=8, learning_rate=0.001, unet_steps=3):
        self.unet_steps = unet_steps
        super().__init__(batch_size, number_of_class, kernel_size, depth, learning_rate)

    def convolution_step(self, data, depth, size=3, train=True):
        hidden = tf.layers.conv2d(data, depth, size, (1, 1), padding="SAME")
        hidden = tf.layers.batch_normalization(hidden, center=True, scale=False, momentum=0.9, training=train)
        return tf.nn.relu(hidden)

    def unet_step(self, data, depth, ending=1, upsample_dims=None, train=True):
        hidden = self.convolution_step(data, depth, train=train)
        hidden = self.convolution_step(hidden, depth, train=train)
        
        if ending == 1:
            return tf.layers.max_pooling2d(hidden, 2, 2, padding="SAME"), hidden
        elif ending == 0:
            return tf.image.resize_nearest_neighbor(hidden, upsample_dims)
        else:
            return hidden

    def forward(self, data, train=True):
        hidden = data
        hidden_connect = []

        for step in range(self.unet_steps):
            hidden, hidden_copy = self.unet_step(hidden, self.depth * (2**step), ending=1, train=train)
            hidden_connect.append(hidden_copy)
        
        for step in range(self.unet_steps-1, -1, -1):
            step_copy = hidden_connect[step]
            hidden = self.unet_step(hidden, self.depth * (2**step), ending=0,
                                    upsample_dims=tf.shape(step_copy)[1:3], train=train)
            hidden = tf.concat([hidden, step_copy], axis=3)

        hidden = self.unet_step(hidden, self.depth, ending=-1, train=train)
        hidden = self.convolution_step(hidden, self.number_of_class, size=1, train=train)
        return hidden

