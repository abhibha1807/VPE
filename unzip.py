# import os
# import zipfile

# def unzip_all_zip_files(directory):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.zip'):
#                 zip_path = os.path.join(root, file)
#                 folder_name = os.path.splitext(file)[0]
#                 folder_path = os.path.join(root, folder_name)
                
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     zip_ref.extractall(folder_path)
#                 os.remove(zip_path)
    
#     print("All zip files extracted successfully!")

# # Example usage
# directory = '/path/to/directory'
# unzip_all_zip_files(directory)

import zipfile

def unzip_file(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Example usage
zip_file_path = '/Users/abhibhagupta/Desktop/CMU/VPE/db/GTSRB/00000.zip'
extract_path = '/Users/abhibhagupta/Desktop/CMU/VPE/db/GTSRB/0000_.zip'
unzip_file(zip_file_path, extract_path)

