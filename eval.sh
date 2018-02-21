#!/bin/bash

for file in ./models/*.npy
do
    filename=$(basename "$file")
    filename="${filename%.*}"
    python main.py --phase=eval --model_file="${file}" > "${filename}.txt"
done
exit 0
