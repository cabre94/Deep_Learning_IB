#! /bin/bash
#$ -N o_100_Alex
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
python ej_10_AlexNet.py -lr 1e-3 -e 100 -bs 256 --dataset 'cifar100'
python ej_10_AlexNet.py -lr 1e-4 -e 100 -bs 256 --dataset 'cifar100'
