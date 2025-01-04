import sys

rate_configuration_name = sys.argv[1]
filename_out = sys.argv[2]

if rate_configuration_name == 'baseline':
    print("Chose baseline name")
else:
    raise ValueError("Invalid configuration name")

with open(filename_out, 'w') as file:
    file.write('testing lalala')

file.close()