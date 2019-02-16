import sys
import os
from scipy import misc
from HelperMethods import evaluate_folder_accuracy, get_coordinates_from_xml
import matplotlib.pyplot as plt

def prepare_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        top = path + '/'
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

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
    path = 'D:\\BP\\detect_test1\\'
    img_path = 'F:\\splited_dataset\\test\\'
    out = 'D:\\bad\\'
    # s = 5
    # n = 25

    prepare_folder(out)
    # acc_list = list()
    # for i in range(s, n):

    accuracy, results = evaluate_folder_accuracy(path, img_path)
    for result in results:
        img = misc.imread(img_path + result[0].replace('xml', 'jpg'), mode="RGB")
        coords = get_coordinates_from_xml(img_path + result[0])
        res = get_img_coords(img, coords)
        plt.imshow(res)
        plt.savefig('{}{}_{}.jpg'.format(out, str(result[1]), result[0]))
    # acc_list.append(accuracy)
    #     print('{}: {}'.format(i, accuracy))

    # for x in range(len(acc_list)):
    #     print('{}: {}'.format(s+x, acc_list[x]))

    return 0


if __name__ == "__main__":
    sys.exit(main())

