import os
import sys
import numpy as np
from shutil import copyfile
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-x', '--xml-path', help='Path to folder where are page xml.',
        default='D:/xml', required=False)
    parser.add_argument(
        '-i', '--img-path', help='Path to folder where are page images.',
        default='F:/impact_dataset', required=False)
    parser.add_argument(
        '-o', '--output-path', help='Path to output folder.', default='D:/BP/output', required=False)
    parser.add_argument(
        '-q', '--quantity', help='Quantity of data.', default=-1, required=False)
    args = parser.parse_args()
    return args


def find_copy_files(img_list, xml_list, img_path, xml_path, output_path, out_names):
    while len(xml_list) > 0:
        index = 0
        if len(xml_list) > 1:
            index = np.random.randint(0, len(xml_list) - 1)
        xml_name = xml_list[index]
        if xml_name in out_names:
            print('Already in')
        img_name = xml_name.replace('.xml', '.jpg')
        xml_list.remove(xml_name)
       
        if img_name in img_list:
            copyfile(img_path + '\\' + img_name, output_path + '\\' + img_name)
            copyfile(xml_path + '\\' + xml_name, output_path + '\\' + xml_name)
        

def main():
    args = parse_arguments()

    xml_names = os.listdir(args.xml_path)
    xml_names = [filename for filename in xml_names if filename.endswith('.xml')]
    img_names = os.listdir(args.img_path)
    img_names = [filename for filename in img_names if filename.endswith('.jpg')]
    out_names = os.listdir(args.output_path)
    wanted = int(args.quantity)
    if wanted < 0:
        wanted = len(xml_names)

    x = 0
    while True:
        if len(xml_names) == 0:
            break
        find_copy_files(img_names, xml_names, args.img_path, args.xml_path, args.output_path, out_names)
        x += 1
        print('Completed {}/{}'.format(x, wanted))
    return 0


if __name__ == '__main__':
    sys.exit(main())