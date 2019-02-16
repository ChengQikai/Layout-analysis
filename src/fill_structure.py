import os
from shutil import copyfile


path = 'F:\\impact_struct'
images_path = 'F:\\impact_dataset'

folders = os.listdir(path)
for folder in folders:
    xml_files = os.listdir('{}/{}'.format(path, folder))
    for xml_file in xml_files:
        img_name = xml_file.replace('.xml', '.jpg')
        copyfile('{}/{}'.format(images_path, img_name), '{}/{}/{}'.format(path, folder, img_name))

