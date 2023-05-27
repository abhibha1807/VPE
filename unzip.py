# import os
# import zipfile

# def unzip_all_zip_files(directory):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.zip'):
#                 zip_path = os.path.join(root, file)
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     zip_ref.extractall(root)
#                 print(f"File '{file}' extracted successfully!")

# # Example usage
# directory = '/path/to/directory'
# unzip_all_zip_files(directory)

import shutil

def move_file(source_file, target_directory):
    shutil.move(source_file, target_directory)

# Example usage
source_file = '/path/to/source/file.txt'
target_directory = '/path/to/target/directory'
move_file(source_file, target_directory)
