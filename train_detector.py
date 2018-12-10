import sys
import os
import argparse
from scipy import misc
import numpy as np
import tensorflow as tf
from SegmentationModel import SegmentationModel
import matplotlib.pyplot as plt


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Model name', default='', required=False)
    parser.add_argument(
        '-i', '--images', help='Path to input image', default='D:\BP\data\inputs', required=False)
    parser.add_argument(
        '-l', '--labels', help='Path to labels', default='D:\BP\data\labels', required=False)
    parser.add_argument(
        '-b', '--batch-size', help='Batch size', default=24, required=False)
    parser.add_argument(
        '-is', '--image-size', help='Image size', default=300, required=False)
    parser.add_argument(
        '-ms', '--max-samples', help='Image size', default=1028, required=False)
    parser.add_argument(
        '-s', '--steps', help='Number of steps', default=10000, required=False)
    args = parser.parse_args()
    return args


class TrainModel:
    def __init__(self, batch_size, img_size=300, model_name=''):
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_name = model_name
        self.model = SegmentationModel(batch_size)
        self.images_list = None
        self.labels_list = None

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
                    img = img/255.0 - 0.5
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
        for i in range(self.steps):
            batch_data, batch_labels = self.get_batch()



def main():
    args = parse_argument()
    trainer = TrainModel(int(args.batch_size), int(args.image_size), int(args.steps), args.model)
    trainer.load_data(args.images, args.labels, max_samples=args.max_samples)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
