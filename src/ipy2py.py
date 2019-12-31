import json
import os
import sys

file_name = sys.argv[1]

if not os.path.exists(file_name):
    print(f"The file {file_name} doesn't exist")
else:
    with open(file_name) as ipynb_file:
        src = json.load(ipynb_file)

        for cell in src["cells"]:
            if cell["cell_type"] == "code":
                for line in cell["source"]:
                    print(line, end='')
            print()
