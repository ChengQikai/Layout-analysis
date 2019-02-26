
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
        '-i', '--input-path', help='Path to input image', default='./data/smaller', required=False)
    parser.add_argument(
        '-b', '--batch-size', help='Batch size', default=24, required=False)
    parser.add_argument(
        '-is', '--image-size', help='Image size', default=300, required=False)
    parser.add_argument(
        '-s', '--steps', help='Number of steps', default=10000, required=False)
    parser.add_argument(
        '-o', '--output', help='Path to output folder', default='.', required=True)
    args = parser.parse_args()
    return args


class TrainModel:
    def __init__(self, batch_size, img_size=300, steps=10000, model_path='', output="."):
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_path = model_path
        self.model = Model(batch_size, depth=32, number_of_class=3, unet_steps=3)
        self.images_list = None
        self.labels_list = None
        self.weights_list = None
        self.test_images_list = None
        self.test_labels_list = None
        self.test_weights_list = None
        self.steps = steps
        self.output = output
        plt.figure(dpi=300)

    def load_part_data(self, images_path, labels_path, weights_path, images_names, labels_names, weights_names):
        images_list = []
        labels_list = []
        weights_list = []

        for img_name in images_names:
            if img_name in labels_names and img_name in weights_names:
                img = misc.imread(images_path + '/' + img_name, mode='RGB')
                if len(img.shape) < 2:
                    continue
                if img.shape[0] > self.img_size and img.shape[1] > self.img_size:
                    label = misc.imread(labels_path + '/' + img_name, mode='L')
                    weights_map = misc.imread(weights_path + '/' + img_name, mode='L')
                    img = img / 255.0 - 0.5
                    images_list.append(img)
                    labels_list.append(label)
                    weights_list.append(weights_map)

        return images_list, labels_list, weights_list

    def load_data(self, path):
        self.test_images_list = []
        self.test_labels_list = []

        train_images_path = path + '/train/inputs'
        train_labels_path = path + '/train/labels'
        train_weights_path = path + '/train/weights_map'

        test_images_path = path + '/test/inputs'
        test_labels_path = path + '/test/labels'
        test_weights_path = path + '/test/weights_map'

        names = os.listdir(train_images_path)
        labels_names = os.listdir(train_labels_path)
        weights_names = os.listdir(train_weights_path)

        test_names = os.listdir(test_images_path)
        test_label_names = os.listdir(test_labels_path)
        test_weights_names = os.listdir(test_weights_path)

        self.images_list, self.labels_list, self.weights_list = self.load_part_data(
            train_images_path, train_labels_path, train_weights_path, names, labels_names, weights_names)

        self.test_images_list, self.test_labels_list, self.test_weights_list = self.load_part_data(
            test_images_path, test_labels_path, test_weights_path, test_names, test_label_names, test_weights_names)

    def get_batch(self, size=None, is_test=False):
        batch_data = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        batch_labels = np.zeros([self.batch_size, self.img_size, self.img_size])
        batch_weights = np.zeros([self.batch_size, self.img_size, self.img_size])

        if size is None:
            size = self.batch_size

        for i in range(size):
            if is_test:
                img_index = np.random.randint(0, len(self.test_images_list) - 1)
                img = self.test_images_list[img_index]
                label = self.test_labels_list[img_index]
                weights_map = self.test_weights_list[img_index]
            else:
                img_index = np.random.randint(0, len(self.images_list) - 1)
                img = self.images_list[img_index]
                label = self.labels_list[img_index]
                weights_map = self.weights_list[img_index]

            x = np.random.randint(0, img.shape[1] - self.img_size)
            y = np.random.randint(0, img.shape[0] - self.img_size)
            batch_data[i] = img[y:y + self.img_size, x:x + self.img_size]
            batch_labels[i] = label[y:y + self.img_size, x:x + self.img_size]
            batch_weights[i] = weights_map[y:y + self.img_size, x:x + self.img_size]
        return batch_data, batch_labels, batch_weights

    def train(self):
        loss_list = list()
        accuracy_list = list()
        accuracy_list_test = list()

        with tf.Session(graph=self.model.graph) as session:
            tf.global_variables_initializer().run()

            if self.model_path != '':
                print(self.model_path)
                self.model.saver.restore(session, tf.train.latest_checkpoint(self.model_path))
                print('ReStored')

            self.model.saver.save(session, self.output + '/model/model')

            for step in range(self.steps + 1):
                batch_data, batch_labels, batch_weights = self.get_batch()
                feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                             self.model.tf_train_mode: True, self.model.tf_train_weights: batch_weights}

                _, loss = session.run([self.model.optimizer, self.model.loss], feed_dict=feed_dict)
                print('{}: {}'.format(step, loss))
                loss_list.append(loss)

                if step != 0 and step % 1000 == 0:
                    validation_size = self.batch_size

                    batch_data, batch_labels, batch_weights = self.get_batch(size=validation_size, is_test=False)
                    feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                                 self.model.tf_train_mode: False, self.model.tf_train_weights: batch_weights}

                    output = session.run([self.model.output_map], feed_dict=feed_dict)
                    output = output[0]

                    output_seg = HelperMethods.get_segmentation_map(output)
                    accuracy = HelperMethods.pixel_accuracy(output_seg, batch_labels)
                    accuracy_list.append(accuracy)

                    print("Step {}: Loss {} \nTrain Accuracy {}%".format(step, loss, accuracy * 100))
                    HelperMethods.save_loss_graph(step, loss_list, self.output)

                    os.makedirs('{}/overview/train/{}'.format(self.output, step))

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
                        plt.savefig('{}/overview/train/{}/{}'.format(self.output, step, i))

                    # batch_data, batch_labels = self.get_batch(size=validation_size, is_test=True)
                    # feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                    #              self.model.tf_train_mode: False}
                    # output = session.run([self.model.output_map], feed_dict=feed_dict)
                    # output = output[0]
                    # output_seg = HelperMethods.get_segmentation_map(output)
                    # accuracy = HelperMethods.pixel_accuracy(output_seg, batch_labels)
                    # accuracy_list_test.append(accuracy)
                    # print("Test: Accuracy {}%".format(accuracy * 100))
                    # plt.clf()
                    # plt.plot(accuracy_list, color="r", label="train")
                    # plt.plot(accuracy_list_test, color="b", label="test")
                    # plt.savefig('{}/accuracy/{}.jpg'.format(self.output, step))
                    #
                    # os.makedirs('{}/overview/test/{}'.format(self.output, step))
                    # output_seg = np.stack((output_seg,) * 3, axis=-1)
                    # np.putmask(output_seg, output_seg == (1, 1, 1), (255, 0, 0))
                    # np.putmask(output_seg, output_seg == (2, 2, 2), (0, 255, 0))
                    #
                    # for i in range(validation_size):
                    #     plt.clf()
                    #     ax = plt.subplot(1, 3, 1)
                    #     ax.set_title("Input")
                    #     plt.imshow(batch_data[i] + 0.5)
                    #     ax = plt.subplot(1, 3, 2)
                    #     ax.set_title("Label")
                    #     label = batch_labels[i]
                    #     label = np.array(label, dtype="int")
                    #     label = np.stack((label,) * 3, axis=-1)
                    #     np.putmask(label, label == (1, 1, 1), (255, 0, 0))
                    #     np.putmask(label, label == (2, 2, 2), (0, 255, 0))
                    #     plt.imshow(label)
                    #     ax = plt.subplot(1, 3, 3)
                    #     ax.set_title("Output")
                    #     plt.imshow(output_seg[i])
                    #     plt.savefig('{}/overview/test/{}/{}'.format(self.output, step, i))

                    self.model.saver.save(session, self.output + '/model/model', global_step=step,
                                          write_meta_graph=False)


def prepare_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
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
                         args.output)
    trainer.load_data(args.input_path)
    prepare_folder(args.output + '/loss')
    prepare_folder(args.output + '/accuracy')
    prepare_folder(args.output + '/overview')
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
