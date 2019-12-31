#!/bin/bash

(cd sudoku-solver && make)
./flask-setup.sh

docker build -t visudoku:latest .
