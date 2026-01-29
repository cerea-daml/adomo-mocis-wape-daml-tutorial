# Data assimilation applied to a shallow water model: practical sessions

##### Tobias Finn, CEREA, [tobias.finn@enpc.fr](mailto:tobias.finn@enpc.fr)

##### Sibo Cheng, CEREA, [sibo.cheng@enpc.fr](mailto:sibo.cheng@enpc.fr)

[![DOI](https://zenodo.org/badge/590754159.svg)](https://zenodo.org/doi/10.5281/zenodo.10478752)

During these sessions, you will apply two classical data assimilation methods 
to a shallow water model. The objective for you is to better understand these 
methods, figure out their practical implementations and identify their key parameters.

The Advanced tutorial, which consists of training a surrogate neural network model and performing parameter calibration using 3D-Var, will not be considered in the course evaluation.

These practical sessions, originally designed by Vivien Mallet and Alban Farchi , are part of the 
[data assimilation course](http://cerea.enpc.fr/HomePages/bocquet/teaching/) 
by Marc Bocquet.

## Installation

You have two options to execute the `tutorial_TODO.ipynb` and `Advanced_tutorial_TODO.ipynb`: you could either use Google Colab or you could install it locally.


### Google Colab

To use the tutorial_TODO notebook with the shared resources from Google Colab, you need a Google account.

Click on: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fAoNsHphKx8NXIwi1b6SdUnERxcG1ATq?usp=sharing)

To use the Advanced_tutorial_TODO notebook with the shared resources from Google Colab, 

Click on: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Kg9wAgJLeJZC4qVDBroObfa152Mdm6S1?usp=sharing)

For that notebook, you have only reading rights. To execute the notebook, you need to save the notebook in your own Google Drive: `File -> Save a copy in Drive`, a new tab will open with the copied notebook.

After accessing your own copy, you have to connect to the runtime. On the top right, there is a button `connect`, when you click on it, you instantiate your own runtime. Afterwards, you're ready to go with the notebook. All needed `git clone` and python `import` are performed for you in the notebook.

### Local installation

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

    $ jupyter-notebook tutorial_TODO.ipynb

