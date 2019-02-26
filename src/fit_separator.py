import sys
from scipy import misc
import matplotlib.pyplot as plt
import lxml.etree as ET


def get_line_coords(path):
    coords = []
    root = ET.parse(path).getroot()

    for region in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion'):
        region_coordinates = region.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords')
        points_string = region_coordinates.attrib['points'].split(' ')
        poly_points = []

        for point_string in points_string:
            point = point_string.split(',')
            poly_points.append((int(point[1]), int(point[0])))

    return coords

def main():
    path = '../data/smaller2/test/labels/322471.png'
    xml_path = '../data/322471.png'
    img = misc.imread(path)
    plt.imshow(img)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())