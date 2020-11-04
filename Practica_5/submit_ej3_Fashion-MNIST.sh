#! /bin/bash
#$ -N o_3_F
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpushort
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 1e-2 -e 100 -bs 256
#python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 1e-3 -e 100 -bs 256
#python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 5e-4 -e 100 -bs 256
#python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 1e-4 -e 100 -bs 256
#python ej_3_Fashion-MNIST.py -lr 1e-2 -rf 5e-5 -e 100 -bs 256
