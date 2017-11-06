#works for the kaggle dataset. We also need cookies if have to download on SCC
import io
from zipfile import ZipFile
import csv
import requests

# The direct link to the Kaggle data set
data_url = 'https://www.kaggle.com/c/passenger-screening-algorithm-challenge/download/stage1_aps.tar.gz'

# The local path where the data set is saved.
local_filename = "test.csv.zip"

# Kaggle Username and Password
kaggle_info = {'UserName': "gvikram", 'Password': "*******"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get(data_url)

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info)

# Writes the data to a local file one chunk at a time.
f = open(local_filename, 'wb')
for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
    if chunk: # filter out keep-alive new chunks
        f.write(chunk)
f.close()

#c = ZipFile(local_filename)
