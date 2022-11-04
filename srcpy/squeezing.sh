#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"
#
# for d in 0.1 0.2 0.5 1 1.5
# do
#     for g in 0.1 0.2 0.4 0.8 1.2
#     do
#         run2 -t 48:00 -n "sq32" "python squeezing.py -n 3 -m 2 --bmin 0 --bmax 20 --bnum 100 -d $d -g $g"
#         run2 -t 48:00 -n "sq43" "python squeezing.py -n 4 -m 3 --bmin 0 --bmax 20 --bnum 100 -d $d -g $g"
#         run2 -t 48:00 -n "sq34" "python squeezing.py -n 3 -m 4 --bmin 0 --bmax 20 --bnum 100 -d $d -g $g"
#         # run2 -t 05:00 -n "sq35" "python squeezing.py -n 3 -m 5 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq36" "python squeezing.py -n 3 -m 6 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq45" "python squeezing.py -n 4 -m 5 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq53" "python squeezing.py -n 5 -m 3 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq54" "python squeezing.py -n 5 -m 4 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq55" "python squeezing.py -n 5 -m 5 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq44" "python squeezing.py -n 4 -m 4 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#         # run2 -t 05:00 -n "sq33" "python squeezing.py -n 3 -m 3 --bmin 0 --bmax 20 --bnum 500 -d $d -g $g"
#     done
# done

run2 -t 96:00 -n "sq21" "python squeezing.py -n 2 -m 1 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq22" "python squeezing.py -n 2 -m 2 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq23" "python squeezing.py -n 2 -m 3 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq32" "python squeezing.py -n 3 -m 2 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq33" "python squeezing.py -n 3 -m 3 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq34" "python squeezing.py -n 3 -m 4 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq43" "python squeezing.py -n 4 -m 3 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq44" "python squeezing.py -n 4 -m 4 --bmax 6 --bnum 100"
run2 -t 96:00 -n "sq45" "python squeezing.py -n 4 -m 5 --bmax 6 --bnum 100"

run2 -t 96:00 -n "sq21e" "python squeezing.py -n 2 -m 1 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq22e" "python squeezing.py -n 2 -m 2 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq23e" "python squeezing.py -n 2 -m 3 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq32e" "python squeezing.py -n 3 -m 2 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq33e" "python squeezing.py -n 3 -m 3 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq34e" "python squeezing.py -n 3 -m 4 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq43e" "python squeezing.py -n 4 -m 3 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq44e" "python squeezing.py -n 4 -m 4 --bmax 6 --bnum 100 -e"
run2 -t 96:00 -n "sq45e" "python squeezing.py -n 4 -m 5 --bmax 6 --bnum 100 -e"
