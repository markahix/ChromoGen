#!/bin/bash
#SBATCH -t 14-0
#SBATCH --job-name ChromoGen_AB
#SBATCH --mem=128GB
#SBATCH -q express
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ChromoGen_AB.log
#SBATCH -e ChromoGen_AB.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gi1632@wayne.edu

module load cuda/11.2

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda not found, installing Miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "export PATH=\${PATH}:\${HOME}/miniconda/bin" >> \${HOME}/.bashrc
    export PATH=\${PATH}:\${HOME}/miniconda/bin
    conda init bash
else
	source /wsu/home/gi/gi16/gi1632/.bashrc
	#conda init bash
fi

# Display conda version
echo "Using conda version:" 
echo | which conda

# Check if the conda environment exists
if conda env list | grep -q 'ChromoGen'
then
    echo "Conda environment ChromoGen exists"
else
    echo "Conda environment not found"
	exit
fi

# Activate the conda environment
conda activate ChromoGen

# Check if the conda environment was activated successfully
if [[ $(conda env list | grep '*' | awk '{print $1}') != "ChromoGen" ]]
then
    echo "Failed to activate conda environment ChromoGen"
    exit
else
	echo "'ChromoGen' conda environment activated successfully."
fi

python main.py --load_pretrained --pretrained_path ./data/models/gpt_pre_rl_gdb13.pt --do_eval --dataset_path ./data/gdb/gdb13/GDB13_Subset-AB.smi --tokenizer Char --tokenizer_path ./data/tokenizers/gdb13ScaffoldCharTokenizer.json --rl_epochs 250 --rl_size 250000 --batch_size 256 --eval_steps 25