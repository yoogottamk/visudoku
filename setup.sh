#!/bin/bash

(cd sudoku-solver && make)
make

docker build -t visudoku:latest .
