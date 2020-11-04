#! /bin/bash
#$ -N MNIST
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

# Ejercicio 6, con dos capas de 20 neuronas
python Conv_MNIST.py -lr 1e-4 -rf 1e-3 -do 0 -e 200 -bs 512
