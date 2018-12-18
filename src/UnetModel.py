from SegmentationModel import SegmentationModel
import tensorflow as tf


class UnetModel(SegmentationModel):
    def convolution_step(self, data, depth, size=3):
        hidden = tf.layers.conv2d(data, depth, size, (1, 1), padding="SAME")
        hidden = tf.layers.batch_normalization(hidden, center=True, scale=False)
        return tf.nn.relu(hidden)

    def unet_step(self, data, depth, ending=1, upsample_dims=None):
        hidden = self.convolution_step(data, depth)
        hidden = self.convolution_step(hidden, depth)
        if ending == 1:
            return tf.layers.max_pooling2d(hidden, 2, 2, padding="SAME"), hidden
        elif ending == 0:
            return tf.image.resize_nearest_neighbor(hidden, upsample_dims)
        else:
            return hidden

    def forward(self, data):
        hidden = data
        hidden_connect = []

        for step in range(3):
            hidden, hidden_copy = self.unet_step(hidden, self.depth * (2**step), ending=1)
            hidden_connect.append(hidden_copy)
        for step in range(2, -1, -1):
            step_copy = hidden_connect[step]
            hidden = self.unet_step(hidden, self.depth * (2**step), ending=0, upsample_dims=tf.shape(hidden)[1:3] * 2)
            hidden = tf.concat([hidden, step_copy], axis=3)

        hidden = self.unet_step(hidden, self.depth, ending=-1)
        hidden = self.convolution_step(hidden, 2, size=1)
        return hidden

