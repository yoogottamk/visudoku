default:
	jupyter nbconvert --to=script src/visudoku.ipynb --stdout | sed '/^\s*#/d' | cat -s > app/visudoku.py
	[ -d app/uploads ] || mkdir app/uploads
