default:
	jupyter nbconvert --to=script src/visudoku.ipynb --stdout | sed '/^\s*#/d' | sed 's/^\(\s*\)plt.*/\1pass/' | cat -s > flask-app/visudoku.py
	[ -d flask-app/uploads ] || mkdir flask-app/uploads
