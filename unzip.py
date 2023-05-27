import zipfile

def unzip_file(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Example usage
zip_file_path = '/path/to/file.zip'
extract_path = '/path/to/extract/location'
unzip_file(zip_file_path, extract_path)
