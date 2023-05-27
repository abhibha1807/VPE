import os
import shutil

def move_folder_contents(source_folder, target_folder):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Move all contents of the source folder to the target folder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        if os.path.isfile(item_path):
            shutil.move(item_path, target_folder)
        elif os.path.isdir(item_path):
            shutil.move(item_path, target_folder)

# Example usage
source_folder = '/path/to/source/folder'
target_folder = '/path/to/target/folder'
move_folder_contents(source_folder, target_folder)
