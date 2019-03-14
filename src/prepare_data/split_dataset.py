import os
from random import randint

book_folder_path = 'D:\\structured_smaller2'
output_path = 'D:\\smaller_splited_dataset2'

folders = os.listdir(book_folder_path)
folders_length = len(folders)
train_dataset_length = (folders_length//3) * 2
train_books_names = []

while train_dataset_length > 0:
    index = randint(0, train_dataset_length)
    book = folders[index]
    train_books_names.append(book)
    folders.remove(book)
    train_dataset_length -= 1

test_dataset_names = folders

for train_dataset in train_books_names:
    os.system('python prepare_folders.py -i "{}/{}" -x "{}/{}" -o "{}/train"'
              .format(book_folder_path, train_dataset, book_folder_path, train_dataset, output_path))
for test_dataset in test_dataset_names:
    os.system('python prepare_folders.py -i "{}/{}" -x "{}/{}" -o "{}/test"'
              .format(book_folder_path, test_dataset, book_folder_path, test_dataset, output_path))
