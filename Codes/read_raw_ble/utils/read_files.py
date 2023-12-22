import os 
import numpy as np
from datetime import datetime

def remove_prefix(text:str, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def remove_suffix(text:str, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text

def parse_filename(filename:str, file_prefix:str, file_suffix:str='.npy'):
    """Extract date and time info in the file name"""
    base_name = os.path.basename(filename)
    date_str = remove_prefix(filename, file_prefix)
    date_str = remove_suffix(date_str, file_suffix)
    try:
        file_datetime = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
        return file_datetime
    except ValueError:
        return None

def find_latest_file_with_prefix_and_suffix(folder_path:str, file_prefix:str, file_suffix:str='.npy'):
    """find the latest file with given prefix"""
    latest_file = None
    latest_date = None

    for file in os.listdir(folder_path):
        if file.startswith(file_prefix) and file.endswith(file_suffix):
            file_date = parse_filename(file, file_prefix, file_suffix)
            if file_date is not None and (latest_date is None or file_date > latest_date):
                latest_date = file_date
                latest_file = file
    
    if latest_file is not None:
        return latest_file
    else: 
        raise FileNotFoundError(f"Corresponding file starts with \"{file_prefix}\" and ends with \"{file_suffix}\" not found")
        

def test():
    print(find_latest_file_with_prefix_and_suffix("mytest", file_prefix="offset-"))

if __name__  == '__main__':
    test()
    