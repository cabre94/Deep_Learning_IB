#! /bin/bash
#$ -N o_1_VGG
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu@compute-6-7.local
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=10G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

#python ej_01.py -lr 1e-3 -rf 3e-3 -e 100 -bs 512 --dogs_cats 'small'
python ej_1.py -lr 1e-4 -rf 0 -e 200 -bs 8
#python ej_1.py -lr 1e-2 -rf 3e-3 -e 100 -bs 32
