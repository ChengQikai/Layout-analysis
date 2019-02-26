import sys
import os
from HelperMethods import evaluate_symetric_best_dice, get_coordinates_from_xml
def main():
    detected_path = '../dbscan/'
    original_path = '../data/test/'
    names = os.listdir(detected_path)
    names = [filename for filename in names if filename.endswith('.xml')]


    for name in names:
        first_coords = get_coordinates_from_xml(detected_path + name)
        second_coords = get_coordinates_from_xml(original_path + name)
        accurracy = evaluate_symetric_best_dice(first_coords, second_coords)
        print(accurracy)

    return 0


if name == '__main__':
    sys.exit(main())