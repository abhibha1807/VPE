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

import os
import shutil

def move_files_and_delete_directory(source_directory):
    target_directory = os.path.dirname(source_directory)  # Get the parent directory

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_directory, file)
            shutil.move(source_file_path, target_file_path)

    shutil.rmtree(source_directory)  # Remove the source directory

    print("Files moved successfully and directory deleted!")

# Example usage
source_directory = '/path/to/dir1/dir2'
move_files_and_delete_directory(source_directory)

