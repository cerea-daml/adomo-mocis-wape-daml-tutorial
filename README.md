# Data assimilation applied to a shallow water model: practical sessions

Alban Farchi, CEREA, [alban.farchi@enpc.fr](mailto:alban.farchi@enpc.fr)

During these sessions, you will apply two classical data assimilation methods 
to a shallow water model. The objective for you is to better understand these 
methods, figure out their practical implementations and identify their key parameters.

## installation

Clone the repertory with HTTPS (cloning with SSH is not allowed outside ENPC).

    $ git clone https://gitlab.enpc.fr/alban.farchi/tp-data-assimilation.git

Create a dedicated anaconda environment for the sessions:

    $ conda env create -f environment.yml

Activate the newly created environment:

    $ conda activate da-tp-sw

Open the notebook and follow the instructions:

    $ jupyter-notebook tp-shallow-water.ipynb
