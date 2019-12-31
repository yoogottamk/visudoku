#!/bin/bash

python3 src/ipy2py.py src/visudoku.ipynb | sed '/^\s*#/d' | sed 's/^\(\s*\)plt.*/\1pass/' | sed 's/.*matplotlib.*//g' | cat -s > flask-app/visudoku.py
[ -d flask-app/uploads ] || mkdir flask-app/uploads
