#!/bin/bash

## give a name to your job
#SBATCH --job-name=JOBNAME1

## your contact email
#SBATCH --mail-user=roman.koshkin@oist.jp

## number of cores for your simulation,
## for serial job array it is always 1
#SBATCH --ntasks=1

#SBATCH --partition=compute

## how much memory per core
#SBATCH --mem-per-cpu=32G

## maximum time for your simulation, in DAY-HOUR:MINUTE:SECOND
#SBATCH --time=0-12:0:0
#SBATCH -c 2


##python Ca_Buffer_GHK_ser.py $SLURM_ARRAY_TASK_ID 10
## $1 means that we pass the first argument (passed into this bash scipt) into the python script
## python Ca_Buffer_GHK_ser.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $1

python cluster.py $1 $2 $3 1800 1800 &

