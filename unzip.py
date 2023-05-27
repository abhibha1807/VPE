import os
import zipfile

def unzip_all_zip_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)
    
    print("All zip files extracted successfully!")

# Example usage
directory = '/Users/abhibhagupta/Desktop/CMU/VPE/db/GTSRB_Test'
unzip_all_zip_files(directory)