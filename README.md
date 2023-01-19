# Data assimilation applied to a shallow water model: practical sessions

##### Alban Farchi, CEREA, [alban.farchi@enpc.fr](mailto:alban.farchi@enpc.fr)

During these sessions, you will apply two classical data assimilation methods 
to a shallow water model. The objective for you is to better understand these 
methods, figure out their practical implementations and identify their key parameters.

These practical sessions, originally designed by Vivien Mallet, are part of the 
[data assimilation course](http://cerea.enpc.fr/HomePages/bocquet/teaching/) 
by Marc Bocquet.

## Installation

Install conda, for example through [miniconda](https://docs.conda.io/en/latest/miniconda.html) or through [mamba](https://mamba.readthedocs.io/en/latest/installation.html).

Clone the repertory:

    $ git clone git@github.com:cerea-daml/adomo-mocis-wape-daml-tutorial.git

Go to the repertory. Once there, create a dedicated anaconda environment for the sessions:

    $ conda env create -f environment.yaml

Activate the newly created environment:

    $ conda activate tutorial

[Optional] Update the environment:

    $ conda update --all

[Optional] Test the environment (this may take up to one minute):

    $ python test_import.py

Open the notebook (e.g. with Jupyter) and follow the instructions:

    $ jupyter-notebook tutorial.ipynb

