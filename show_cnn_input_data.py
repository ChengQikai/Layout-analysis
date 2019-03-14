import os
from scipy import misc
import matplotlib.pyplot as plt

path = './data/sg_vertical/train'

img_files = os.listdir('{}/{}'.format(path, 'inputs'))
for img_file in img_files:
    img = misc.imread('{}/{}/{}'.format(path, 'inputs', img_file),mode="RGB")
    label = misc.imread('{}/{}/{}'.format(path, 'labels', img_file))
    map = misc.imread('{}/{}/{}'.format(path, 'weights_map', img_file))
    fig = plt.figure(dpi=300)
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.imshow(map)
    plt.savefig('./show/{}'.format(img_file))
    plt.close(fig)
