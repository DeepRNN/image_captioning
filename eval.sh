#!/bin/bash

for file in ./models/*.npy
do
    filename=$(basename "$file")
    filename="${filename%.*}"
    for value in {1..5}
    do
        python main.py --phase=val --model_file="$file" --beam_size=$value > "${filename}_${value}.txt"
    done
done
exit 0
