import os
from shutil import copyfile
import random


path = 'F:\\impact_struct'
out = 'F:\\structured_smaller\\'

folders = os.listdir(path)
for folder in folders:
    xml_files = os.listdir('{}/{}'.format(path, folder))
    xml_files = [file for file in xml_files if file.endswith('.xml')]
    random.shuffle(xml_files)
    xml_files = xml_files[:30]
    new_folder = ''.join(e for e in folder if e.isalnum())
    if os.path.exists(out + new_folder):
        continue
    os.mkdir(out + new_folder)

    for xml_file in xml_files:
        img_name = xml_file.replace('.xml', '.jpg')
        copyfile('{}/{}/{}'.format(path, folder, xml_file), '{}{}/{}'.format(out, new_folder, xml_file))
        copyfile('{}/{}/{}'.format(path, folder, img_name), '{}{}/{}'.format(out, new_folder, img_name))



