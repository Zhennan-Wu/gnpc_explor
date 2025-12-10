#!/bin/bash

# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gnpc

# 3. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name gnpc --display-name "Python (gnpc)"

echo "=== Environment 'gnpc' setup complete! ==="
echo "Use 'conda activate gnpc' to activate and select 'Python (gnpc)' in Jupyter."
