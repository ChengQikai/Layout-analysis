import tensorflow as tf
import sys
from UnetModel import UnetModel as Model
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import os
import scipy.ndimage.measurements as mnts


def get_output(outputs):
    output_max = np.maximum(outputs[:, :, :, 0], outputs[:, :, :, 1])
    output_max = np.maximum(output_max, outputs[:, :, :, 2])
    x = np.array(outputs[:, :, :, 0] == output_max, dtype="int") * 0
    y = np.array(outputs[:, :, :, 1] == output_max, dtype="int") * 1
    z = np.array(outputs[:, :, :, 2] == output_max, dtype="int") * 2
    return x + y + z


def one_img(session, model, input_path, output_path):
    img = misc.imread(input_path)
    input_data = np.array([(img / 255.0) - 0.5])
    feed_dict = {model.tf_train_dataset: input_data, model.tf_train_mode: False}
    output = session.run([model.output_map], feed_dict=feed_dict)

    output_seg = get_output(output[0])[0]
    plot_seg = output_seg.copy()
    plot_seg = np.stack((plot_seg,) * 3, axis=-1)
    np.putmask(plot_seg, plot_seg == (1, 1, 1), (255, 0, 0))
    np.putmask(plot_seg, plot_seg == (2, 2, 2), (0, 255, 0))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.subplot(1, 4, 2)
    plt.imshow(plot_seg)

    np.putmask(output_seg, output_seg == 2, 0)
    spectral_data = output_seg.astype(bool)
    distance = ndi.distance_transform_edt(spectral_data)
    print(distance.shape, spectral_data.shape)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=spectral_data)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=spectral_data)
    plt.subplot(1, 4, 3)
    plt.imshow(labels)
    structure = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    bboxSlices = {}
    for i in range(1, labels.max() + 1):
        B = labels.copy()
        B[B != i] = 0

        bboxSlices[i] = mnts.find_objects(mnts.label(labels, structure=structure)[0])

    img_cpy = img.copy()

    plt.subplot(1, 4, 4)
    label = Image.new("L", (img_cpy.shape[1], img_cpy.shape[0]), 0)
    for key in bboxSlices.keys():
        for rec in bboxSlices[key]:
            line_points = []
            width_slice = rec[0]
            height_slice = rec[1]
            line_points.append((height_slice.start, width_slice.start))
            line_points.append((height_slice.start, width_slice.stop))
            line_points.append((height_slice.stop, width_slice.stop))
            line_points.append((height_slice.stop, width_slice.start))
            line_points.append((height_slice.start, width_slice.start))
            ImageDraw.Draw(label).line(line_points, width=2, fill=key)
        break
    label = np.array(label)

    for y in range(img_cpy.shape[1]):
        for x in range(img_cpy.shape[0]):
            if label[x][y] == 1:
                img_cpy[x][y] = (255, 0, 0)
    plt.imshow(img_cpy)

    plt.savefig('{}/{}'.format(output_path,input_path.split('/')[-1]))


def main():
    print('Main')
    model = Model(24, depth=32, number_of_class=3)
    base_path = '../data/split/test/inputs'
    with tf.Session(graph=model.graph) as session:
        model.saver.restore(session, tf.train.latest_checkpoint('../experiments/unet/05/model/'))
        names = os.listdir(base_path)
        count = len(names)
        actual = 0
        for name in names:
            one_img(session, model, '{}/{}'.format(base_path, name), '../clusters')
            actual += 1
            print('{}/{}'.format(actual, count))
    return 0


if __name__ == '__main__':
    sys.exit(main())

