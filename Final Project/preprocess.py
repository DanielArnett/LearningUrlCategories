import csv
import os
import math
import numpy as np
import string

source_data_path = './final_set.csv'
output_path = './final_set_preprocessed.csv'
try:
    os.unlink(output_path)
    print('Deleted ' + output_path)
except:
    print('No file to delete.')

class_names = np.array(['Adult', 'Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Kids', 'News', 'Recreation',
               'Reference', 'Science', 'Shopping', 'Society', 'Sports'])

# Increase to speed up processing and use more RAM
batch_size = 10000


# def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
#     csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
#     for row in csv_reader:
#         yield [unicode(cell, 'utf-8') for cell in row]


def open_csv_as_numpy():
    csv_as_list = list()
    with open(source_data_path, 'rb') as fin:
        reader = csv.reader(fin, dialect=csv.excel)
        csv_as_list = list(reader)
    # reader = unicode_csv_reader(open(source_data_path))
    #     csv_as_list = list(reader)
    return np.asarray(csv_as_list)


csv_as_ndarray = open_csv_as_numpy()
new_shape = [batch_size, class_names.shape[0]]
formatted_ndarray = np.chararray(new_shape)
formatted_ndarray[:] = '0'
num_of_batches = int(math.ceil(csv_as_ndarray.shape[0] / float(batch_size)))
fout = open(output_path, 'ab')
for i in range(num_of_batches - 1):
    lo = i*batch_size
    if i < num_of_batches - 1:
        hi = (i+1)*batch_size
    elif i == num_of_batches - 1:
        hi = csv_as_ndarray.shape[0]
    output_buffer = formatted_ndarray[:hi-lo].astype('U256')
    if i == 0:
        output_buffer[0] = np.array(class_names).astype('U256')
    for j in range(0, class_names.shape[0]):
        output_buffer[csv_as_ndarray[lo:hi,2] == class_names[j], j] = '1'
    # csv_as_ndarray = np.resize(csv_as_ndarray, new_shape)
    # formatted_ndarray = formatted_ndarray.astype(csv_as_ndarray.dtype)
    try:
        output_buffer = np.concatenate((csv_as_ndarray[lo:hi, :2], output_buffer), axis=1)
    except UnicodeDecodeError:
        printable = set(string.printable)
        for index, url in enumerate(csv_as_ndarray[lo:hi, 1]):
            try:
                url.encode('utf-8')
            except UnicodeDecodeError:
                print('Problematic string: \"' + url)
                new_url = filter(lambda x: x in printable, url)
                print('Using string: ' + new_url)
                csv_as_ndarray[lo:hi, 1][index] = new_url
        output_buffer = np.concatenate((csv_as_ndarray[lo:hi, :2], output_buffer), axis=1)
    np.savetxt(fout, output_buffer, fmt='%s', delimiter=",")

