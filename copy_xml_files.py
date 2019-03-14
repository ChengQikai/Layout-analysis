import os
import sys
from shutil import copyfile

orig_path = 'D:\\DATA\\impact_parsed_xml'
new_path = 'D:\\smaller_splited_dataset2\\train'
xml_path = 'D:\\splited_xml\\train'

files = os.listdir(orig_path)
xml_files = [filename for filename in files if filename.endswith('.xml')]
out_files = os.listdir(new_path)
out_files = [filename for filename in out_files if filename.endswith('.xml')]

for out in out_files:
    if out in xml_files:
        copyfile('{}/{}'.format(orig_path, out), '{}/{}'.format(xml_path, out))
