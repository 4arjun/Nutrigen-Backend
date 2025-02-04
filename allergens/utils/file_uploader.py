import os

def save_to_temp_file(file_data):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    temp_directory = os.path.join(script_directory, '../../temp')
    temp_directory = os.path.abspath(temp_directory)

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    
    temp_file_path = os.path.join(temp_directory, file_data.name)

    with open(temp_file_path, 'wb') as temp_file:
        for chunk in file_data.chunks():
            temp_file.write(chunk)

    print(f"File saved to {temp_file_path}")
    return read_temp_file(temp_file_path)

def read_temp_file(file_path):
    with open(file_path, 'rb') as f:
        contents = f.read()
        return contents
