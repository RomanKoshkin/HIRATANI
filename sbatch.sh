#!/bin/bash

SIMULATION_T=10

for U in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for HAGA in 1 0
    do
        for asym in 1 0
        do
           sleep 2
           python cluster.py $HAGA $asym $U $SIMULATION_T &
        done
    done
done
