# Digit recognition

Some images are present in `train`.
To get more examples, (you should) run the `data_augment.py` file

After generating them, you can run `gen_data.py` to convert the data and labels in a format suitable for
training models on. The files will be stored as `X.pickle` and `y.pickle`

Now, after generating X and y, we are ready for training. You can run `train.py`, which creates `knn.model` which is used by `visudoku.ipynb`
