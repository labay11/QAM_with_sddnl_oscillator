#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

run2 -t 120:00 -c 4 -n "evo1" "julia evolution.jl -n 3 -m 3 --tmax 5000 --tnum 50000 -e 0.47412171491586075 -g 0.8"
run2 -t 120:00 -c 4 -n "evo2" "julia evolution.jl -n 3 -m 3 --tmax 5000 --tnum 50000 -e 1.3755952741694344 -g 0.6"
run2 -t 120:00 -c 4 -n "evo3" "julia evolution.jl -n 3 -m 3 --tmax 5000 --tnum 50000 -e 1.984379582650444 -g 0.4"


run2 -t 120:00 -c 4 -n "evo4" "julia evolution.jl -n 4 -m 4 --tmax 5000 --tnum 50000 -e 0.5684021044214842 -g 0.15"
run2 -t 120:00 -c 4 -n "evo5" "julia evolution.jl -n 4 -m 4 --tmax 5000 --tnum 50000 -e 1.1423118881998557 -g 0.1"
run2 -t 120:00 -c 4 -n "evo6" "julia evolution.jl -n 4 -m 4 --tmax 5000 --tnum 50000 -e 1.4210869697181017 -g 0.05"
