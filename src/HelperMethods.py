import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def pixel_accuracy(output, label):
    correct = np.sum(np.equal(output, label))
    total = output.shape[0] * output.shape[1] * output.shape[2]
    return correct / total


def save_loss_graph(step, loss, output):
    plt.clf()
    plt.plot(savgol_filter(loss, 75, 3))
    plt.savefig('{}/loss/{}.jpg'.format(output, step))


def get_segmentation_map(outputs):
    output_max = np.maximum(outputs[:, :, :, 0], outputs[:, :, :, 1])
    output_max = np.maximum(output_max, outputs[:, :, :, 2])
    x = np.array(outputs[:, :, :, 0] == output_max, dtype="int") * 0
    y = np.array(outputs[:, :, :, 1] == output_max, dtype="int") * 1
    z = np.array(outputs[:, :, :, 2] == output_max, dtype="int") * 2
    return x + y + z

