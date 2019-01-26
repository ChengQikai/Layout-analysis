import sys
from DocumentAnalyzer import DocumentAnalyzer
from scipy import misc
import matplotlib.pyplot as plt
import skimage.draw as skd
import random


def main():
    input_path = 'C:\\Users\\david\\Desktop\\document_segmentation\\316931.jpg'
    analyzer = DocumentAnalyzer()
    coordinates = analyzer.get_document_paragraphs(input_path)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

