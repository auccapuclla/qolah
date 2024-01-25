import re
from plotVHF import *
import os
import pandas as pd
import gc
import time
from datetime import datetime

def extract_timestamp(filename):
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    match = re.search(timestamp_pattern, filename)
    if match:
        return match.group()
    else:
        return None

def find_match(input_string):
    # pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"
    pattern = r"\d{2}T\d{2}:\d{2}"
    match = re.search(pattern, input_string)
    return match.group() if match else None

def custom_key_to_sort(filename):
    timestamp = datetime.strptime(filename.split('_')[0], '%Y-%m-%dT%H:%M:%S.%f')
    return timestamp.day, timestamp.hour, timestamp.minute, timestamp.second

path_to_files = '/mnt/nas-fibre-sensing/20231115_Cintech_Heterodyne_Phasemeter/'
files = []

for item in os.listdir(path_to_files):
    full_path = os.path.join(path_to_files, item)
    if os.path.isfile(full_path):  # Check if it's a file
        if "01-12" in item:
            files.append(item)


files = sorted(files, key=custom_key_to_sort)
for file in files:
    print(file)

data = pd.DataFrame(columns=['Timestamp', '# Val','Phase Val', 'Indices'])
data_list = []

for filename in files:
    print(f' processing: {filename}')
    file = Path(path_to_files+filename)
    r = get_radius(VHFparser(file, skip=2000))
    filtered_values = r[r < 600]
    indices = np.where(r < 600)[0]
    data_list.append({
        'Timestamp': extract_timestamp(filename),
        '# Val': len(filtered_values),
        'Phase Val': filtered_values.tolist(), 
        'Indices': indices.tolist()
        })
    del r
    gc.collect()
    time.sleep(5)

data = pd.DataFrame(data_list)

data.to_csv(f'./test.csv', index=False)