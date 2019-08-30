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


import sys
import os
import argparse
from scipy import misc
import numpy as np
import tensorflow as tf
from UnetModel import UnetModel as Model
import matplotlib.pyplot as plt
import HelperMethods
import random


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to Model to load', default='', required=False)
    parser.add_argument(
        '-i', '--input-path', help='Path to input image', default='../data/training', required=False)
    parser.add_argument(
        '-b', '--batch-size', help='Batch size', default=8, required=False)
    parser.add_argument(
        '-is', '--image-size', help='Image size', default=300, required=False)
    parser.add_argument(
        '-s', '--steps', help='Number of steps', default=20000, required=False)
    parser.add_argument(
        '-o', '--output', help='Path to output folder', default='./training_output', required=False)
    parser.add_argument(
        '-a', '--batch-sample', default='', required=False)
    args = parser.parse_args()
    return args


class TrainModel:
    def __init__(self, batch_size, img_size=300, steps=10000, model_path='', output=".", batch_sample=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_path = model_path
        self.model = Model(batch_size, depth=32, number_of_class=3, unet_steps=4)
        self.images_list = None
        self.labels_list = None
        self.weights_list = None
        self.names_list = None
        self.train_probabilities_name = []
        self.train_probabilities = []
        self.test_images_list = None
        self.test_labels_list = None
        self.test_weights_list = None
        self.test_names_list = None
        self.steps = steps
        self.output = output
        self.batch_sample = batch_sample

    def load_part_data(self, images_path, labels_path, weights_path, images_names, labels_names, weights_names):
        images_list = []
        labels_list = []
        weights_list = []
        names_list = []

        for img_name in images_names:
            if img_name in labels_names and img_name in weights_names:
                img = misc.imread(images_path + '/' + img_name, mode='RGB')
                if len(img.shape) >= 2 and img.shape[0] > self.img_size and img.shape[1] > self.img_size:
                    label = misc.imread(labels_path + '/' + img_name, mode='L')
                    weights_map = misc.imread(weights_path + '/' + img_name, mode='L')
                    img = img / 255.0 - 0.5
                    images_list.append(img)
                    labels_list.append(label)
                    weights_list.append(weights_map)
                    names_list.append(img_name)

        return images_list, labels_list, weights_list, names_list

    def load_data(self, path):
        self.test_images_list = []
        self.test_labels_list = []

        self.train_images_path = path + '/train/inputs'
        self.train_labels_path = path + '/train/labels'
        self.train_weights_path = path + '/train/weights_map'

        self.test_images_path = path + '/test/inputs'
        self.test_labels_path = path + '/test/labels'
        self.test_weights_path = path + '/test/weights_map'

        names = os.listdir(self.train_images_path)
        labels_names = os.listdir(self.train_labels_path)
        weights_names = os.listdir(self.train_weights_path)

        test_names = os.listdir(self.test_images_path)
        test_label_names = os.listdir(self.test_labels_path)
        test_weights_names = os.listdir(self.test_weights_path)

        with open('{}/train/scores'.format(path)) as f:
            prob_string = f.readline()
            prob_string_splited = prob_string.split(';')

            prob_sum = 0
            for img_prob in prob_string_splited:
                img_prob_splited = img_prob.split(',')
                self.train_probabilities_name.append(img_prob_splited[0])
                prob = float(img_prob_splited[1])
                prob_sum += prob
                self.train_probabilities.append(prob)

            self.train_probabilities[0] = 1 - (prob_sum - self.train_probabilities[0])

        self.train_names = []
        for name in names:
            if name in labels_names and name in weights_names:
                self.train_names.append(name)

        self.test_names = []
        for test_name in test_names:
            if test_name in test_label_names and test_name in test_weights_names:
                self.test_names.append(test_name)

    def get_batch(self, size=None, is_test=False):
        batch_data = np.full([self.batch_size, self.img_size, self.img_size, 3], 0.5)
        batch_labels = np.zeros([self.batch_size, self.img_size, self.img_size])
        batch_weights = np.ones([self.batch_size, self.img_size, self.img_size])

        if size is None:
            size = self.batch_size

        actual_size = 0
        while actual_size < size:
            if self.batch_sample == True and actual_size < size // 2:
                img_name = ''
                while  img_name not in self.train_names:
                    img_name = np.random.choice(self.train_probabilities_name, p=self.train_probabilities)
                img_index = self.train_names.index(img_name)
            else:
                img_index = np.random.randint(0, len(self.train_names) - 1)

            name = self.train_names[img_index]
            img = misc.imread(self.train_images_path + '/' + name, mode='RGB')
            try:
                img = np.array(img) / 255.0 - 0.5

                y_size = self.img_size
                if img.shape[0] <= self.img_size:
                    y_size = img.shape[0]
                    y = 0
                else:
                    y = np.random.randint(0, img.shape[0] - y_size)

                x_size = self.img_size
                if img.shape[1] <= self.img_size:
                    x_size = img.shape[1]
                    x = 0
                else:
                    x = np.random.randint(0, img.shape[1] - x_size)

                label = misc.imread(self.train_labels_path + '/' + name, mode='L')
                weights_map = misc.imread(self.train_weights_path + '/' + name, mode='L')

                batch_data[actual_size][0: y_size, 0:x_size] = img[y:y + y_size, x:x + x_size]
                batch_labels[actual_size][0: y_size, 0:x_size] = label[y:y + y_size, x:x + x_size]
                batch_weights[actual_size][0: y_size, 0:x_size] = weights_map[y:y + y_size, x:x + x_size]
                actual_size += 1
            except:
                print('Misc mother fucker')
        return batch_data, batch_labels, batch_weights

    def train(self):
        loss_list = list()
        accuracy_list = list()
        accuracy_list_test = list()

        with tf.Session(graph=self.model.graph) as session:
            tf.global_variables_initializer().run()

            if self.model_path != '' and self.model_path is not None:
                print(self.model_path)
                self.model.saver.restore(session, tf.train.latest_checkpoint(self.model_path))

            self.model.saver.save(session, self.output + '/model/model')

            for step in range(self.steps + 1):
                batch_data, batch_labels, batch_weights = self.get_batch()
                feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                             self.model.tf_train_mode: True, self.model.tf_train_weights: batch_weights}

                _, loss = session.run([self.model.optimizer, self.model.loss], feed_dict=feed_dict)
                print('{}: {}'.format(step, loss))
                loss_list.append(loss)

                if step != 0 and step % 5000 == 0:
                    validation_size = self.batch_size

                    batch_data, batch_labels, batch_weights = self.get_batch(size=validation_size, is_test=False)
                    feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                                 self.model.tf_train_mode: True, self.model.tf_train_weights: batch_weights}

                    output = session.run([self.model.output_map], feed_dict=feed_dict)
                    output = output[0]

                    output_seg = HelperMethods.get_segmentation_map(output)
                    accuracy = HelperMethods.pixel_accuracy(output_seg, batch_labels)
                    accuracy_list.append(accuracy)

                    print("Step {}: Loss {} \nTrain Accuracy {}%".format(step, loss, accuracy * 100))
                    HelperMethods.save_loss_graph(step, loss_list, self.output)

                    output_seg = np.stack((output_seg,) * 3, axis=-1)
                    np.putmask(output_seg, output_seg == (1, 1, 1), (255, 0, 0))
                    np.putmask(output_seg, output_seg == (2, 2, 2), (0, 255, 0))

                    for i in range(validation_size):
                        plt.clf()
                        ax = plt.subplot(1, 3, 1)
                        ax.set_title("Input")
                        plt.imshow(batch_data[i] + 0.5)
                        ax = plt.subplot(1, 3, 2)
                        ax.set_title("Label")
                        label = batch_labels[i]
                        label = np.array(label, dtype="int")
                        label = np.stack((label,) * 3, axis=-1)
                        np.putmask(label, label == (1, 1, 1), (255, 0, 0))
                        np.putmask(label, label == (2, 2, 2), (0, 255, 0))
                        plt.imshow(label)
                        ax = plt.subplot(1, 3, 3)
                        ax.set_title("Output")
                        plt.imshow(output_seg[i])
                        plt.savefig('{}/overview/{}/{}'.format(self.output, step, i))

                    self.model.saver.save(session, self.output + '/model/model', global_step=step,
                                          write_meta_graph=False)


def prepare_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        top = path + '/'
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


def main():
    args = parse_argument()
    trainer = TrainModel(int(args.batch_size), int(args.image_size), int(args.steps), args.model,
                         args.output, args.batch_sample == 'True')
    trainer.load_data(args.input_path)
    prepare_folder(args.output + '/loss')
    prepare_folder(args.output + '/overview')
    prepare_folder(args.output + '/model')
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
