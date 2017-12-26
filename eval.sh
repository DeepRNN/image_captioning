#!/bin/bash

for file in ./models/*.npy
do
    filename=$(basename "$file")
    filename="${filename%.*}"
    python main.py --load --model_file="$file" --phase=val > "$filename".txt
done
exit 0
