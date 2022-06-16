#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

for d in 0.1 0.2 0.5 1 1.5
do
    for g in 0.1 0.2 0.4 0.8 1.2
    do
        run2 -t 24:00 -n "sq32" "julia squeezing.jl -n 3 -m 2 --bmin 0 --bmax 100 --bnum 500 -d $d -g $g"
        run2 -t 24:00 -n "sq43" "julia squeezing.jl -n 4 -m 3 --bmin 0 --bmax 100 --bnum 500 -d $d -g $g"
    done
done

run2 -t 2:00 -n "sq33" "julia squeezing.jl -n 3 -m 3 --bmin 0 --bmax 20 --bnum 100"
run2 -t 2:00 -n "sq44" "julia squeezing.jl -n 4 -m 4 --bmin 0 --bmax 20 --bnum 100"
