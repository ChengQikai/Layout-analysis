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

from UnetModel import UnetModel as Model
from scipy import misc
import tensorflow as tf
import numpy as np
import scipy.ndimage.measurements as mnts
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from skimage import measure
from Postprocessing import paragraphs_postprocessing

def get_img_coords(img, coords):
    res = img.copy()
    label_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
    for rect in coords:
        rect.append(rect[0])
        ImageDraw.Draw(label_img).line(rect, width=15, fill=1)
    label_img = np.array(label_img)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if label_img[y][x] == 1:
                res[y][x] = (0, 255, 0)
    return res


class DocumentAnalyzer:
    def __init__(self, scale=0.23, footprint_size=8, model='../experiments/normal/model'):
        self.__scale = scale
        self.__model = Model(8, depth=32, number_of_class=3,  unet_steps=4)
        self.__restore_path = model
        self.__footprint_size = footprint_size

    def get_document_paragraphs(self, document_path, axarr, or_img, line_coords=None, plot_progress=False):
        img, origin_height, origin_width = self.load_img(document_path)
        axarr[0][0].axis('off')
        axarr[0][0].imshow(img)
        probability_maps = self.get_probability_mask(img)
        segmentation_mask = self.get_segmentation_map(probability_maps)
        axarr[0][1].axis('off')
        axarr[0][1].imshow(segmentation_mask)
        clusters_labels = self.label_clustering(segmentation_mask, axarr)
        axarr[1][0].axis('off')
        axarr[1][0].imshow(clusters_labels)
        coordinates = self.get_coordinates(clusters_labels, origin_width, origin_height)
        res_img = get_img_coords(or_img, coordinates)
        axarr[1][1].axis('off')
        axarr[1][1].imshow(res_img)
        coordinates = paragraphs_postprocessing(coordinates)

        return coordinates, origin_width, origin_height

    def get_probability_mask(self, img):
        with tf.Session(graph=self.__model.graph) as session:
            self.__model.saver.restore(session, tf.train.latest_checkpoint(self.__restore_path))
            input_data = np.array([(img / 255.0) - 0.5])
            feed_dict = {self.__model.tf_train_dataset: input_data, self.__model.tf_train_mode: False}
            output = session.run([self.__model.output_map], feed_dict=feed_dict)
            return output[0]

    @staticmethod
    def get_segmentation_map(probabilities):
        output_max = np.maximum(probabilities[:, :, :, 0], probabilities[:, :, :, 1])
        output_max = np.maximum(output_max, probabilities[:, :, :, 2])
        x = np.array(probabilities[:, :, :, 0] == output_max, dtype="int") * 0
        y = np.array(probabilities[:, :, :, 1] == output_max, dtype="int") * 1
        z = np.array(probabilities[:, :, :, 2] == output_max, dtype="int") * 2
        return (x + y + z)[0]

    @staticmethod
    def clustering(segmentation_mask):
        np.putmask(segmentation_mask, segmentation_mask == 2, 0)
        labels = np.zeros(segmentation_mask.shape, dtype="int32")
        coord = np.nonzero(segmentation_mask == 1)

        if len(coord[0]) > 0:
            z = np.zeros((len(coord[0]), 2), dtype="int32")
            z[:, 0] = coord[0]
            z[:, 1] = coord[1]
            dbscan = DBSCAN(eps=3, min_samples=25,algorithm='auto', leaf_size=30, metric='euclidean',)
            label_1d = dbscan.fit_predict(z)
            labels[coord] = label_1d + 1

        return labels

    def get_coordinates(self, cluster_labels, max_width, max_height):
        ratio = 1 / self.__scale
        coordinates = []
        structure = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        bbox_slices = {}
        for i in range(1, cluster_labels.max() + 1):
            B = cluster_labels.copy()
            B[B != i] = 0
            bbox_slices[i] = mnts.find_objects(mnts.label(B, structure=structure)[0])

        for key in bbox_slices.keys():
            for width_slice, height_slice in bbox_slices[key]:
                rectangle_coordinates = list()
                rectangle_coordinates.append((int(height_slice.start * ratio), int(width_slice.start * ratio)))
                rectangle_coordinates.append((int(height_slice.start * ratio), min(int(width_slice.stop * ratio),
                                                                                   max_width - 1)))
                rectangle_coordinates.append((min(int(height_slice.stop * ratio), max_height-1),
                                              min(int(width_slice.stop * ratio), max_width-1)))
                rectangle_coordinates.append((min(int(height_slice.stop * ratio), max_height-1),
                                              int(width_slice.start * ratio)))
                coordinates.append(rectangle_coordinates)

        return coordinates

    def load_img(self, document_path):
        img = misc.imread(document_path, mode="RGB")
        width, height = img.shape[0:2]
        img = misc.imresize(img, self.__scale)
        return img, height, width

    @staticmethod
    def label_clustering(segmentation_mask,axarr):
        np.putmask(segmentation_mask, segmentation_mask == 2, 0)
        axarr[0][2].axis('off')
        axarr[0][2].imshow(segmentation_mask)
        labels = measure.label(segmentation_mask, background=0)
        return labels
