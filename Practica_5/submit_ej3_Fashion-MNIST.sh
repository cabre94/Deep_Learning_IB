#! /bin/bash
#$ -N o_1
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu
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
python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 1e-3 -e 100 -bs 256
python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 5e-4 -e 100 -bs 256
python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 1e-4 -e 100 -bs 256
python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 5e-5 -e 100 -bs 256