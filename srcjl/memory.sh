#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

for r in {1..400}
do
    run2 -t 0:30 -c 1 -n "4j" "julia memory.jl -n 4 --dim 15"
    run2 -t 1:00 -c 1 -n "4j" "julia memory.jl -n 4 --dim 20"
    run2 -t 1:30 -c 1 -n "4j" "julia memory.jl -n 4 --dim 30"
    run2 -t 2:00 -c 1 -n "4j" "julia memory.jl -n 4 --dim 40"
    sleep 0.1
done
