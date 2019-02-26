#!/usr/bin/env bash

# remove all but the latest 5 checkpoints in given modeldir
checkpoints=$(ls -t $1/*ckpt)

echo $checkpoints
one=1

# skip the last skip_num checkpoints
skip_num=5
counter=0
for c in $checkpoints; do
    counter=$[$counter +1];
    skip=$[$counter>$skip_num]
    if [ "$skip" -eq "$one" ]; then
        echo "delete" $c;
        rm $c
    fi
done