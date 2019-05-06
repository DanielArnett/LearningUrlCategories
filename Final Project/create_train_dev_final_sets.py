import csv
from random import shuffle
import math

dev_set_percentage = 20.0
final_set_percentage = 10.0
source_data_path = './data/URL Classification.csv'
output_path_prefix = './'

with open(source_data_path, 'rb') as f:
    reader = csv.reader(f)
    csv_as_list = list(reader)  

shuffle(csv_as_list)
list_size = csv_as_list.__len__()
dev_set_size = int(math.ceil(float(list_size) * dev_set_percentage / 100.0))
final_set_size = int(math.ceil(float(list_size) * final_set_percentage / 100.0))
final_set_list = csv_as_list[:final_set_size]
dev_set_list = csv_as_list[final_set_size:final_set_size + dev_set_size]
train_set_list = csv_as_list[final_set_size + dev_set_size:]
with open(output_path_prefix + 'final_set.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(final_set_list)
with open(output_path_prefix + 'dev_set.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(dev_set_list)
with open(output_path_prefix + 'train_set.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(train_set_list)
print('here')