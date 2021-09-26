#!/bin/bash
#SBATCH -o ./logs/%j.log
THEANO_FLAGS='device=cuda,floatX=float32'

date -d -30days

#python /cluster/home/cug/yl339/adaptation_proj/train_network_with_TL.py
#python /cluster/home/cug/yl339/adaptation_proj/train_network_ours.py
#python /cluster/home/cug/yl339/adaptation_proj/train_networks.py -filter -ADANN
#python /cluster/home/cug/yl339/adaptation_proj/train_networks.py -ADANN

#python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py
#python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py -normalisation
#python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py -ADANN
#python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py -filter
#python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py -filter -normalisation
python /cluster/home/cug/yl339/adaptation_proj/adap_networks.py -filter -ADANN


date -d -30days
