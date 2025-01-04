import sys
import numpy as np

filename_in = sys.argv[1]
filename_out = sys.argv[2]

# Read the npz file
data = np.load(filename_in)

with open(filename_out, 'w') as file:
    file.write('testing')
    file.write('\t')
    file.write('lalala')

file.close()