import sys
from DocumentAnalyzer import DocumentAnalyzer
from scipy import misc
import matplotlib.pyplot as plt
import skimage.draw as skd
import random
import HelperMethods


def show_graph(input_path, coordinates):
    img = misc.imread(input_path, mode="RGB")
    seg_img = img.copy()
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    r = 18
    g = 159
    b = 220
    for rect in coordinates:
        rr, cc = skd.rectangle((rect[0][1], rect[0][0]), (rect[2][1], rect[2][0]))
        seg_img[rr, cc] = (r, g, b)
        r = r + random.randint(10, 160) % 255
        g = g + random.randint(10, 160) % 255
        b = b + random.randint(10, 160) % 255

    plt.subplot(1, 2, 2)
    plt.imshow(seg_img)
    plt.show()


def main():
    input_path = 'C:\\Users\\david\\Desktop\\document_segmentation\\316931'
    filename = input_path.split('\\')[-1]
    analyzer = DocumentAnalyzer()
    coordinates, img_height, img_width = analyzer.get_document_paragraphs(input_path + '.jpg')
    show_graph(input_path + '.jpg', coordinates)
    xml_string = HelperMethods.create_page_xml(coordinates, img_width, img_height, filename)
    with open('test.xml', 'wb') as f:
        f.write(xml_string)
    return 0


if __name__ == "__main__":
    sys.exit(main())

