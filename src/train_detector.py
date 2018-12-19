import sys
import os
import argparse
from scipy import misc
from scipy.signal import savgol_filter
import numpy as np
import tensorflow as tf
from UnetModel import UnetModel as Model
import matplotlib.pyplot as plt


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to Model to load', default='', required=False)
    parser.add_argument(
        '-i', '--images', help='Path to input image', default='./data/inputs', required=False)
    parser.add_argument(
        '-l', '--labels', help='Path to labels', default='./data/labels', required=False)
    parser.add_argument(
        '-b', '--batch-size', help='Batch size', default=24, required=False)
    parser.add_argument(
        '-is', '--image-size', help='Image size', default=300, required=False)
    parser.add_argument(
        '-ms', '--max-samples', help='Image size', default=1028, required=False)
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
        self.model = Model(batch_size, depth=32)
        self.images_list = None
        self.labels_list = None
        self.steps = steps
        self.output = output

    def load_data(self, images_path, labels_path, max_samples=1028):
        self.images_list = []
        self.labels_list = []
        names = os.listdir(images_path)
        label_names = os.listdir(images_path)

        count = 0
        for name in names:
            if count >= max_samples:
                break
            if name in label_names:
                img = misc.imread(images_path + '/' + name)
                if img.shape[0] >= self.img_size and img.shape[1] >= self.img_size:
                    label = misc.imread(labels_path + '/' + name, mode='L')
                    img = img / 255.0 - 0.5
                    self.images_list.append(img)
                    self.labels_list.append(label)
                    count += 1

    def get_batch(self):
        batch_data = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        batch_labels = np.zeros([self.batch_size, self.img_size, self.img_size])
        for i in range(self.batch_size):
            img_index = np.random.randint(0, len(self.images_list) - 1)
            img = self.images_list[img_index]
            label = self.labels_list[img_index]
            x = np.random.randint(0, img.shape[1] - self.img_size)
            y = np.random.randint(0, img.shape[0] - self.img_size)
            batch_data[i] = img[y:y + self.img_size, x:x + self.img_size]
            batch_labels[i] = label[y:y + self.img_size, x:x + self.img_size]
        return batch_data, batch_labels

    def train(self):
        loss_list = list()
        accuracy_list = list()
        with tf.Session(graph=self.model.graph) as session:
            tf.global_variables_initializer().run()
            if self.model_path != '':
                self.model.saver.restore(session, tf.train.latest_checkpoint(self.model_path))
            self.model.saver.save(session, self.output + '/model/segmentaiton-model')
            for step in range(self.steps + 1):
                batch_data, batch_labels = self.get_batch()
                feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                             self.model.tf_train_mode: True}
                _, l = session.run([self.model.optimizer, self.model.loss], feed_dict=feed_dict)
                print('{}: {}'.format(step, l))
                loss_list.append(l)
                if step != 0 and step % 100 == 0:
                    batch_data, batch_labels = self.get_batch()
                    feed_dict = {self.model.tf_train_dataset: batch_data, self.model.tf_train_labels: batch_labels,
                                 self.model.tf_train_mode: False}
                    output = session.run([self.model.output_map], feed_dict=feed_dict)
                    output = output[0]
                    accuracy = self.pixel_accuracy(output[:, :, :, 1], batch_labels)
                    accuracy_list.append(accuracy)
                    print("Step {}: Loss {}, Accuracy {}%".format(step, l, accuracy * 100))
                    self.save_loss_graph(step, loss_list)
                    plt.clf()
                    plt.plot(accuracy_list)
                    plt.savefig('{}/accuracy/{}.jpg'.format(self.output, step))

                    os.mkdir('{}/overview/{}'.format(self.output, step))
                    for i in range(self.batch_size):
                        plt.clf()
                        ax = plt.subplot(1, 4, 1)
                        ax.set_title("Input")
                        plt.imshow(batch_data[i] + 0.5)
                        ax = plt.subplot(1, 4, 2)
                        ax.set_title("Label")
                        label = batch_labels[i]
                        label = np.array(label, dtype="int")
                        label = np.stack((label,) * 3, axis=-1)
                        label[:, :, 0] *= 255
                        plt.imshow(label)
                        ax = plt.subplot(1, 4, 3)
                        ax.set_title("Out Binary")
                        curr_output = output[i][:, :, 1]
                        curr_output = curr_output > 0.5
                        curr_output = np.array(curr_output, dtype="int")
                        curr_output = np.stack((curr_output,) * 3, axis=-1)
                        curr_output[:, :, 0] *= 255
                        plt.imshow(curr_output)
                        ax = plt.subplot(1, 4, 4)
                        ax.set_title("Output Prob")
                        plt.imshow(output[i][:, :, 1])
                        plt.savefig('{}/overview/{}/{}'.format(self.output, step, i))
                    self.model.saver.save(session, self.output + '/model/segmentaiton-model', global_step=step,
                                          write_meta_graph=False)

    def pixel_accuracy(self, outputs, label):
        outputs = np.array(outputs > 0.5, dtype="int")
        correct = np.sum(np.equal(outputs, label))
        total = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]
        return correct / total

    def save_loss_graph(self, step, loss):
        plt.clf()
        plt.plot(savgol_filter(loss, 75, 3))

        plt.savefig('{}/loss/{}.jpg'.format(self.output, step))


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
    trainer.load_data(args.images, args.labels, max_samples=int(args.max_samples))
    prepare_folder(args.output + '/loss')
    prepare_folder(args.output + '/accuracy')
    prepare_folder(args.output + '/overview')
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
