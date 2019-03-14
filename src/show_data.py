import sys
from HelperMethods import get_coordinates_from_xml, evaluate_symetric_best_dice
import os
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np


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


def main():
    img_path = 'D:\\smaller_splited_dataset2\\train'
    orig_xml_path = 'F:\\impact_parsed'
    xml_path = 'D:\\merged_xml_test4'
    out = 'D:\\detected_without\\'

    files = os.listdir(img_path)
    img_files = [filename for filename in files if filename.endswith('.jpg')]
    xml_files = os.listdir(xml_path)
    xml_files = xml_files[500:]

    for xml_file in xml_files:
        img_file = xml_file.replace('.xml', '.jpg')
        if img_file in img_files:
            img = misc.imread(img_path + '\\' + img_file, mode="RGB")
            plt.subplot(1, 2, 1)
            plt.title(img_file)
            coords, _, _ = get_coordinates_from_xml(xml_path + '\\' + xml_file)
            img_rect = get_img_coords(img, coords)
            plt.imshow(img_rect)
            plt.show()
            # coords, _,_ = get_coordinates_from_xml(orig_xml_path + '\\' + xml_file)
            # img_rect = get_img_coords(img, coords)
            # plt.subplot(1, 2, 2)
            # accuracy = evaluate_symetric_best_dice(xml_path + '\\' + xml_file, orig_xml_path + '\\' + xml_file)
            # plt.title(accuracy)
            # plt.imshow(img_rect)
            # plt.show()
            # plt.savefig('{}{}_{}'.format(out, accuracy, img_file))
            # plt.savefig('{}/{}'.format(out, img_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())\


