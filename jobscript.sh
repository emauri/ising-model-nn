#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 24

cd /home/bissstud/Students16/RobertChau/IsingNNParallel/Debug/

for (( i=1; i<25; i++)); do
(
  srun ./IsingNNParallel $i &>> ./outputCartesius.txt
)&
done

wait 