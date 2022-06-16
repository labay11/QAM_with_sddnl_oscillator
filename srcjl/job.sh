#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

for r in {1..100}
do
    run2 -t 36:00 -n "ev32-$r" "julia ev.jl -n 3 -m 2 -r $r"
done
