import sys
from scipy import misc
import matplotlib.pyplot as plt
from DocumentAnalyzer import DocumentAnalyzer
import os
from SeparatorHelperMethods import rotate_data, get_separator_img, find_separators
from HelperMethods import get_line_coords, get_img_coords


def main():
    analyzer = DocumentAnalyzer(scale=1)
    output = '../separator2'
    path = '../data/vertical1/test/'
    inputs_path = path + 'inputs/'
    labels_path = path + 'labels/'
    xmls_path ='../test/test/'

    inputs_names = os.listdir(inputs_path)
    inputs_names = [filename for filename in inputs_names if filename.endswith('.png')]
    labels_names = os.listdir(labels_path)
    labels_names = [filename for filename in labels_names if filename.endswith('.png')]
    xml_names = os.listdir(xmls_path)
    xml_names = [filename for filename in xml_names if filename.endswith('.xml')]

    for input_name in inputs_names:
        con_name = input_name.replace('.png', '.xml')
        if input_name in labels_names and con_name in xml_names:
            img_path = inputs_path + input_name
            path = labels_path + input_name
            xml_path = xmls_path + con_name

            img_o = misc.imread(img_path, mode="RGB")
            img = misc.imread(path)
            line_coords = get_line_coords(xml_path)

            probability_maps = analyzer.get_probability_mask(img_o)
            segmentation_mask = analyzer.get_segmentation_map(probability_maps)

            rotated_img, rotation = rotate_data(img, line_coords)
            rotated_img_o, rotation = rotate_data(img_o, line_coords)
            rotated_img1, rotation1 = rotate_data(segmentation_mask, line_coords)

            sep = find_separators(rotated_img)
            sep_len = len(sep)
            sep1 = find_separators(rotated_img1)
            sep1_len = len(sep1)

            fig = plt.figure(dpi=300)
            plt.subplot(2, 3, 1)
            plt.imshow(img_o)
            plt.subplot(2, 3, 2)
            plt.imshow(segmentation_mask)
            plt.subplot(2, 3, 3)
            plt.title('Ground {}: {}'.format(sep_len, 0))
            plt.imshow(get_separator_img(rotated_img_o.copy(), sep, (0, 255, 0)))
            plt.subplot(2, 3, 4)
            plt.title('CNN {}: {}'.format(sep1_len, 0))
            plt.imshow(get_separator_img(rotated_img_o.copy(), sep1, (0, 255, 0)))

            rotated_img1 = get_separator_img(rotated_img1.copy(), sep1, 2)
            # labels = analyzer.label_clustering(rotated_img1)
            labels = analyzer.clustering(rotated_img1)
            plt.subplot(2, 3, 5)
            plt.imshow(labels)
            coords = analyzer.get_coordinates(labels, rotated_img1.shape[0], rotated_img1.shape[1])
            plt.subplot(2, 3, 6)
            result = get_img_coords(rotated_img_o, coords)
            plt.imshow(result)

            plt.savefig('{}/{}'.format(output, input_name))
            plt.close(fig)
            print('Complete one')

    print("Complete all")
    return 0


if __name__ == "__main__":
    sys.exit(main())