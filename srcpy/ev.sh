#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

for r in {0..99}
do
    run2 -t 36:00 -n "ev33-$r" "python ev.py -n 3 -m 3 -r $r --g1 0.0000001"
    run2 -t 36:00 -n "ev44-$r" "python ev.py -n 4 -m 4 -r $r --g1 0.0000001 --g2max 0.2"
done
