import sys
import argparse
from HelperMethods import evaluate_symetric_best_dice


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
    accuracy = evaluate_symetric_best_dice(args.first_xml, args.second_xml)
    print(accuracy)

    return 0


if __name__ == "__main__":
    sys.exit(main())

