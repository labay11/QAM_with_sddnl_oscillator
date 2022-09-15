#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

run2 -t 48:00 -m 10 -c 4 -n "traj" "python trajectory.py"
