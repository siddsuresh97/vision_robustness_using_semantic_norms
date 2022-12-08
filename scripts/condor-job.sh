#!/usr/bin/bash

export HOME=$PWD
export PATH
sh vision_robustness_using_semantic_norms/Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3
export PATH=$PWD/miniconda3/bin:$PATH

cat > ~/.bashrc <<EOF
export HOME=$PWD
export PATH
export PATH=$PWD/miniconda3/bin:$PATH
EOF
conda init bash
conda create -n alexnet python=3.7
conda activate alexnet
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install deepspeed
#pip install transformers

cp /staging/suresh27/ecoset_leuven_updated.tar.gz .

tar -xvf ecoset_leuven_updated.tar.gz
sleep 7d
