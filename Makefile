default:
	python3 src/ipy2py.py src/visudoku.ipynb | sed '/^\s*#/d' | sed 's/^\(\s*\)plt.*/\1pass/' | cat -s > flask-app/visudoku.py
	[ -d flask-app/uploads ] || mkdir flask-app/uploads
