# Data assimilation applied to a shallow water model: practical sessions

##### Alban Farchi, CEREA, [alban.farchi@enpc.fr](mailto:alban.farchi@enpc.fr)

During these sessions, you will apply two classical data assimilation methods 
to a shallow water model. The objective for you is to better understand these 
methods, figure out their practical implementations and identify their key parameters.

These practical sessions, originally designed by Vivien Mallet, are part of the 
[data assimilation course](http://cerea.enpc.fr/HomePages/bocquet/teaching/) 
by Marc Bocquet.

## Installation

Clone the repertory with HTTPS (cloning with SSH is not allowed outside ENPC).

    $ git clone https://gitlab.enpc.fr/alban.farchi/tp-data-assimilation.git

Go to the repertory. Once there, create a dedicated anaconda environment for the sessions:

    $ conda env create -f environment.yml

Activate the newly created environment:

    $ conda activate da-tp-sw

[Optional] Update the environment:

    $ conda update --all

[Optional] Test the environment (this may take up to one minute):

    $ python test_import.py

Open the notebook and follow the instructions:

    $ jupyter-notebook tp-shallow-water.ipynb
