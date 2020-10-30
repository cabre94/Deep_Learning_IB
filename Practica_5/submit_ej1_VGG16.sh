#! /bin/bash
#$ -N o_1_VGG
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=8G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

#python ej_01.py -lr 1e-3 -rf 3e-3 -e 100 -bs 512 --dogs_cats 'small'
python ej_1.py -lr 1e-2 -rf 3e-3 -e 100 -bs 512
