# import standard modules
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns
from tqdm import trange
from IPython.display import HTML

# for plot customisation
sns.set_context('notebook')
sns.set_style('darkgrid')
plt.rc('axes', linewidth=1)
plt.rc('axes', edgecolor='k')
plt.rc('figure', dpi=300)
palette = sns.color_palette('deep')

# import custom shallow water model package
import shallow_water_model as swm

# list of parameters
Nx = 101
dx = 1
dt = 0.03
Q = 0.1
g = 9.81
Nt = 500
h_anom = 1.05

# create the model
sw_model = swm.ShallowWaterModel(Nx, dx, dt, Q, g)

# create a driver
def forecast_driver(model, state, Nt): 
    for t in trange(Nt, desc='running forward model'):
        model.forward(state)

print('It looks like everything is fine!')

