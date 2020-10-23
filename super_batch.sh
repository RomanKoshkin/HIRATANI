#!/bin/bash


for U in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for HAGA in 1 0
    do
        for asym in 1 0
        do
           echo $HAGA $asym $U
           sleep 2
           sbatch bt_serial.sh $HAGA $asym $U
        done
    done
done