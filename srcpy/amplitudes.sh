#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

run2 -t 05:00 -n "amp32" "python amplitudes.py -n 3 -m 2 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp43" "python amplitudes.py -n 4 -m 3 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp34" "python amplitudes.py -n 3 -m 4 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp35" "python amplitudes.py -n 3 -m 5 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp36" "python amplitudes.py -n 3 -m 6 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp45" "python amplitudes.py -n 4 -m 5 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp53" "python amplitudes.py -n 5 -m 3 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp54" "python amplitudes.py -n 5 -m 4 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp55" "python amplitudes.py -n 5 -m 5 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp44" "python amplitudes.py -n 4 -m 4 --bmin 0 --bmax 20 --bnum 500"
run2 -t 05:00 -n "amp33" "python amplitudes.py -n 3 -m 3 --bmin 0 --bmax 20 --bnum 500"
