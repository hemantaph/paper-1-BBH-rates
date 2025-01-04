import sys
import numpy as np

rate_configuration = sys.argv[1]
if rate_configuration == 'baseline':
    print("Chose baseline name")
else:
    raise ValueError("Invalid configuration name")

# npz file to be created
filename_out = sys.argv[2]
