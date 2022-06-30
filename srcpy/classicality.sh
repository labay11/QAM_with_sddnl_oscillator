#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

for g in 0.05 0.2 0.4
do
  for r in {0..50}
  do
      run2 -t 1:00 -n "ev32-$r" "python classicallity.py -n 3 -m 2 -j $r -g $g"
      run2 -t 1:00 -n "ev33-$r" "python classicallity.py -n 3 -m 3 -j $r -g $g"
      run2 -t 1:00 -n "ev22-$r" "python classicallity.py -n 2 -m 2 -j $r -g $g"
      run2 -t 1:00 -n "ev43-$r" "python classicallity.py -n 4 -m 3 -j $r -g $g"
      run2 -t 1:00 -n "ev42-$r" "python classicallity.py -n 4 -m 2 -j $r -g $g"
      run2 -t 1:00 -n "ev44-$r" "python classicallity.py -n 4 -m 4 -j $r -g $g"
  done
done
