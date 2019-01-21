#!/bin/bash
python3 predict.py 0 \
& python3 predict.py 1 \
& python3 predict.py 2 \
& python3 predict.py 3 \
& python3 predict.py 4 \
& python3 predict.py 5 \
& python3 predict.py 6 \
& python3 predict.py 7
