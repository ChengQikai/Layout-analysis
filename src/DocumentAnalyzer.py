from UnetModel import UnetModel as Model
from scipy import misc
import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import scipy.ndimage.measurements as mnts


class DocumentAnalyzer:
    def __init__(self, scale=0.09):
        self.__scale = scale
        self.__model = Model(24, depth=32, number_of_class=3)
        self.__restore_path = '../experiments/unet/05/model/'

    def get_document_paragraphs(self, document_path):
        img, origin_height, origin_width = self.load_img(document_path)
        probability_maps = self.get_probability_mask(img)
        segmentation_mask = self.get_segmentation_map(probability_maps)
        clusters_labels = self.clustering(segmentation_mask)
        coordinates = self.get_coordinates(clusters_labels, origin_width, origin_height)
        return coordinates

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
        boolean_map = segmentation_mask.astype(bool)
        distance = ndi.distance_transform_edt(boolean_map)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                    labels=boolean_map)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=boolean_map)
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
            bbox_slices[i] = mnts.find_objects(mnts.label(cluster_labels, structure=structure)[0])

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
            break
        return coordinates

    def load_img(self, document_path):
        img = misc.imread(document_path, mode="RGB")
        width, height = img.shape[0:2]
        img = misc.imresize(img, self.__scale)
        return img, height, width

