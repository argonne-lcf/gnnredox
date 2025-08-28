To run the codes here, first clone the repository and cd inside the repo dir.

Next, create a Python virtual environment or a new Conda environment. Then install the Python packages as shown below.

**Python version used**: Python 3.11.9 

If you are using Conda:

```
# Note, exporting the PYTHONNOUSERSITE variable is required if you are on an HPC or ALCF Polaris or Sophia
# Setting the PYTHONNOUSERSITE variable ensures that your new conda environment will not see any HPC conda environment (if there are any)
export PYTHONNOUSERSITE=1
conda create -p ./conenv python=3.11.9
```

Package installation (same for both virtual environment or Conda environment):

```
pip install -r requirements.txt

# torch packages cannot be installed through a requirements file, so they have to be installed manually
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

Since most of the codes here are in a Jupyter notebook, you have to setup a Jupyter kernel. 

First activate your new environment. If it is virtual environment, source it using `source path/to/your/venv/bin/activate`

If it is a Conda environement, use `conda activate path/to/your/conda_env`

```
python -m ipykernel install --user --name=conenv --display-name "aBeautifulName"
```

