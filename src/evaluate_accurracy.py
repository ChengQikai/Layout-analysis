import sys
import argparse
from HelperMethods import get_maps, get_coordinates_from_xml, symmetric_best_dice


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fx', '--first-xml', help='Path to first xml', required=True)
    parser.add_argument(
        '-sx', '--second-xml', help='Path to second xml', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    first_xml_coordinates, width, height = get_coordinates_from_xml(args.first_xml)
    second_xml_coordinates, _, _ = get_coordinates_from_xml(args.second_xml)
    first_maps, second_maps = get_maps(first_xml_coordinates, second_xml_coordinates, width, height)
    accuracy = symmetric_best_dice(first_maps, second_maps)
    print(accuracy)

    return 0


if __name__ == "__main__":
    sys.exit(main())

