import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import lxml.etree as ET
from PIL import Image, ImageDraw
import os

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


def best_dice(a, b):
    sum_a = 0

    if len(a) == 0:
        return 0

    for ai in a:
        max_value = 0
        for bj in b:
            eq = np.sum(np.equal(ai, bj) * ai)
            top_val = 2 * eq
            bottom_val = np.sum(ai) + np.sum(bj)
            curr_val = top_val/bottom_val
            max_value = max(max_value, curr_val)
        sum_a += max_value

    return sum_a / len(a)


def symmetric_best_dice(ground_truth, result):
    return min(best_dice(ground_truth, result), best_dice(result, ground_truth))


def get_maps(coordinates, ground_truth_coordinates, image_width, image_height):
    ground_truth_maps = []
    results_maps = []

    for poly_points in coordinates:
        img = Image.new("L", (image_width, image_height), 0)
        poly = np.array(poly_points)
        poly = list(map(tuple, poly))
        ImageDraw.Draw(img).polygon(poly, fill=1)
        results_maps.append(np.array(img))

    for poly_points in ground_truth_coordinates:
        img = Image.new("L", (image_width, image_height), 0)
        poly = np.array(poly_points)
        poly = list(map(tuple, poly))
        ImageDraw.Draw(img).polygon(poly, fill=1)
        ground_truth_maps.append(np.array(img))

    return ground_truth_maps, results_maps


def get_coordinates_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()

    width = None
    height = None

    for page in root:
        width = int(page.attrib['imageWidth'])
        height = int(page.attrib['imageHeight'])

    coordinates = []

    for region in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion'):
        region_coordinates = region.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords')
        points_string = region_coordinates.attrib['points'].split(' ')
        poly_points = []

        for point_string in points_string:
            point = point_string.split(',')
            poly_points.append((int(point[1]), int(point[0])))

        coordinates.append(poly_points)

    return coordinates, width, height


def create_page_xml(coordinates, width, height, filename):
    root = ET.Element("PcGts")
    root.set('xmlns', 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15')
    page_element = ET.SubElement(root, 'Page',
                                 {"imageFilename": filename, "imageWidth": str(width), "imageHeight": str(height)})
    region_index = 1
    for text_region in coordinates:
        text_region_element = ET.SubElement(page_element, 'TextRegion', {"id": "r{}".format(region_index)})
        coords = ''
        for point in text_region:
            coords += '{},{} '.format(point[1], point[0])
        coords = coords.strip()
        ET.SubElement(text_region_element, 'Coords', {"points": coords})
        region_index += 1

    return ET.tostring(root, encoding='utf-8',  pretty_print=True)


def evaluate_symetric_best_dice(first_xml, second_xml):
    first_xml_coordinates, width, height = get_coordinates_from_xml(first_xml)
    second_xml_coordinates, _, _ = get_coordinates_from_xml(second_xml)
    first_maps, second_maps = get_maps(first_xml_coordinates, second_xml_coordinates, width, height)
    return symmetric_best_dice(first_maps, second_maps)


def evaluate_folder_accuracy(first_folder, second_folder):
    first_folder_names = os.listdir(first_folder)
    second_folder_names = os.listdir(second_folder)

    first_folder_names = [filename for filename in first_folder_names if filename.endswith('.xml')]
    second_folder_names = [filename for filename in second_folder_names if filename.endswith('.xml')]

    sum = 0
    count = 0
    result = []

    for name in first_folder_names:
        if name in second_folder_names:
            accuracy = evaluate_symetric_best_dice('{}{}'.format(first_folder, name),
                                                   '{}/{}'.format(second_folder, name))
            print(accuracy)
            result.append((name, accuracy))
            sum += accuracy
            count += 1

    if count == 0:
        return 0

    return sum / count, sorted(result, key=lambda x: x[1])

