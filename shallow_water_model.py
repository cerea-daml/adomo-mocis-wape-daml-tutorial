r"""Python implementation of a unidimensional Shallow Water model.

The model describes the time evolution of the water height \(h(x)\)
and the horizontal velocity \(v(x)\) in a fixed-length domain. The 
model equations read:
\[
    \frac{\partial h}{\partial t} + \frac{\partial(hu)}{\partial x} = 0,\\
    \frac{\partial(hu)}{\partial t} + \frac{\partial(hu^2)}{\partial x} + gh\frac{\partial h}{\partial x} = 0.
\]

The boundary conditions are the following:

1. on the left, a constant inflow \(Q=hu\);
2. on the right, a homogeneous Neumann condition for \(h\) and \(u\), 
   where fluxes are consequently determined by the state of the system 
   along the boundary.

The equations are solved by the `shallow_water_forward` function using 
numerical schemes detailed by, e.g., [Honnorat, 2007](https://tel.archives-ouvertes.fr/tel-00273318) 
and originally implemented by [V. Mallet](mailto:vivien.mallet@inria.fr). 
The `ShallowWaterModel` class defines a user-friendly object-oriented 
interface to the `shallow_water_forward` function and can be used as follows.

    model = ShallowWaterModel()
    state = model.new_state_crenel()
    for t in range(Nt):
        model.forward(state)
"""

import numpy as np
from numba import njit

class ShallowWaterState:
    r"""Container class for a Shallow Water model state.

    A model state is a couple of numpy arrays `(h, u)` of size `Nx`
    (the number of grid points) containing the values of \(h(x)\) and
    \(u(x)\) in the domain. For simplicity, each model state also stores
    a couple of numpy arrays `(fh, fu)` of size `Nx+1`. These arrays
    are used as temporary storage during the model integration to 
    compute the flux values for \(h(x)\) and \(u(x)\).

    Attributes
    ----------
    h : numpy array of size Nx
        The values of \(h(x)\).
    u : numpy array of size Nx
        The values of \(u(x)\).
    fh : numpy array of size Nx+1
        The flux values for \(h(x)\).
    fu : numpy array of size Nx+1
        The flux values for \(u(x)\).
    """

    def __init__(self, Nx):
        """Init the model state."""
        self.h = np.ones(Nx)
        self.u = np.zeros(Nx)
        self.fh = np.zeros(Nx+1)
        self.fu = np.zeros(Nx+1)

class ShallowWaterEnsembleState:
    r"""Container class for an ensemble of Shallow Water model states.

    See the documentation for the `ShallowWaterState` class.

    Attributes
    ----------
    Ne : int
        Ensemble size.
    h : numpy array of shape (Ne, Nx)
        The values of \(h(i, x)\).
    u : numpy array of shape (Ne, Nx)
        The values of \(u(i, x)\).
    fh : numpy array of size Nx+1
        The flux values for \(h(x)\).
    fu : numpy array of size Nx+1
        The flux values for \(u(x)\).
    """

    def __init__(self, Ne, Nx):
        """Init the ensemble of model states."""
        self.Ne = Ne
        self.h = np.ones((Ne, Nx))
        self.u = np.zeros((Ne, Nx))
        self.fh = np.zeros(Nx+1)
        self.fu = np.zeros(Nx+1)

class ShallowWaterModel:
    """Implementation of the Shallow Water model.

    Attributes
    ----------
    Nx : integer, optional
        The number of grid points.
    dx : real, optional
        The size of the grid cell.
    dt : real, optional
        The integration step.
    Q : real, optional
        The constant inflow on the left.
    g : real, optional
        The acceleration due to gravity.
    """

    def __init__(self, Nx=100, dx=1, dt=0.03, Q=0.1, g=9.81):
        """Init the model."""
        self.Nx = Nx
        self.dx = dx
        self.dt = dt
        self.Q = Q
        self.g = g

    def new_state_crenel(self, h_anom=1.05):
        r"""Create a new model state.

        The water height \(h(x)\) is a crenel: \(h(x)=1\) everywhere but
        in the center of the domain, where \(h(x)=h_a\).
        The horizontal velocity \(u(x)\) is null.

        Arguments
        ---------
        h_anom : real, optional
            The water height at the center \(h_a\).

        Returns
        -------
        state : ShallowWaterState instance 
            The new model state.
        """
        ic = (self.Nx-1) // 2
        i_min = max(ic-5, 0)
        i_max = min(ic+5, self.Nx-1)
        state = ShallowWaterState(self.Nx)
        state.h[i_min:i_max+1] = h_anom
        return state

    def new_ensemble_crenel(self, 
                            Ne,
                            mean_h_anom=0, 
                            std_h_anom=0.02, 
                            debias=True):
        """Create an ensemble of model states.

        Each ensemble member is built as in the `ShallowWaterModel.new_state_crenel` 
        method but with a random value for `h_anom`, drawn from the 
        normal distribution with mean `mean_h_anom` and standard deviation 
        `std_h_anom`.

        Arguments
        ---------
        Ne : int
            Ensemble size
        mean_h_anom : real, optional
            The average water anomaly in the ensemble.
        std_h_anom : real, optional
            The standard deviation of water anomaly in the ensemble.
        debias : boolean, optional
            If True, the sample mean of `h_anom` is corrected to be
            exactly `mean_h_anom`.

        Returns
        -------
        ensemble : ShallowWaterEnsembleState instance
            The new ensemble of model states.
        """
        ic = (self.Nx-1) // 2
        i_min = max(ic-5, 0)
        i_max = min(ic+5, self.Nx-1)
        h_anom = np.random.randn(Ne)
        if debias:
            h_anom -= h_anom.mean()
        h_anom = mean_h_anom + std_h_anom * h_anom
        ensemble = ShallowWaterEnsembleState(Ne, self.Nx)
        for i in range(Ne):
            ensemble.h[i, i_min:i_max+1] = h_anom[i]
        return ensemble

    def forward(self, state):
        """Perform one integration step.

        Uses the `shallow_water_forward` function to integrate the model 
        state.

        Arguments
        ---------
        state : ShallowWaterState instance
            The model state to integrate.
        """
        shallow_water_forward(self.Nx, 
                              self.dx, 
                              self.dt, 
                              self.Q, 
                              self.g, 
                              state.h, 
                              state.u, 
                              state.fh, 
                              state.fu)

    def forward_ensemble(self, ensemble):
        """Perform one integration step.

        Uses the `shallow_water_forward` function to integrate the model 
        states.

        Arguments
        ---------
        ensemble : ShallowWaterEnembleState instance
            The ensemble of model states to integrate.
        """
        for i in range(ensemble.Ne):
            shallow_water_forward(self.Nx, 
                                  self.dx, 
                                  self.dt, 
                                  self.Q, 
                                  self.g, 
                                  ensemble.h[i], 
                                  ensemble.u[i], 
                                  ensemble.fh, 
                                  ensemble.fu)

@njit(cache=True)
def shallow_water_forward(Nx, dx, dt, Q, g, h, u, fh, fu):
    r"""Perform one integration of the Shallow Water model.

    The integration is performed in-place and relies on
    `numba` for efficiency.

    Arguments
    ---------
    Nx : integer
        The number of grid points.
    dx : real
        The size of the grid cell.
    dt : real
        The integration step.
    Q : real 
        The constant inflow on the left.
    g : real 
        The acceleration due to gravity.
    h : numpy array of size Nx
        The values of \(h(x)\).
    u : numpy array of size Nx
        The values of \(u(x)\).
    fh : numpy array of size Nx+1
        Temporary storage of the flux values for \(h(x)\).
    fu : numpy array of size Nx+1
        Temporary storage of the flux values for \(u(x)\).
    """

    # left boundary: constant inflow
    fh[0], fu[0] = _compute_flux_hll(g, h[0], h[0], Q/h[0], u[0])

    # fluxes inside the domain
    for i in range(Nx-1):
        fh[i+1], fu[i+1] = _compute_flux_hll(g, h[i], h[i+1], u[i], u[i+1])

    # right boundary: free
    fh[Nx], fu[Nx] = _compute_flux_hll(g, h[Nx-1], h[Nx-1], u[Nx-1], u[Nx-1])

    # update h and u
    for i in range(Nx):
        h[i] += dt * (fh[i]-fh[i+1]) / dx
        u[i] += dt * (fu[i]-fu[i+1]) / dx

@njit(cache=True)
def _compute_flux_hll(g, hl, hr, ul, ur):
    """Auxiliary function for the model integration."""

    # phase speed of the wave
    cl = np.sqrt(g*hl)
    cr = np.sqrt(g*hr)
    clr = cl + cr

    # height at the interface
    ulr = ul - ur
    h_tmp = 0.5 * (hl+hr) * (1+0.5*ulr/clr)
    if h_tmp <= min(hl, hr):
        tmp = 0.5*clr + 0.25*ulr
        h_interface = tmp**2/g
    elif h_tmp >= max(hl, hr):
        gl = np.sqrt(0.5*g*(h_tmp+hl)/(h_tmp*hl))
        gr = np.sqrt(0.5*g*(h_tmp+hr)/(h_tmp*hr))
        h_interface = (hl*gl+hr*gr+ulr) / (gl+gr)
    else:
        h_interface = h_tmp

    # wave velocity
    pl = np.sqrt(0.5*h_interface*(h_interface+hl)) / hl if h_interface > hl else 1
    sl = ul - cl*pl
    if sl >= 0:
        return _compute_flux(g, hl, ul)
    pr = np.sqrt(0.5*h_interface*(h_interface+hr)) / hr if h_interface > hr else 1
    sr = ur + cr*pr
    if sr <= 0:
        return _compute_flux(g, hr, ur)

    # flux
    fhl, ful = _compute_flux(g, hl, ul)
    fhr, fur = _compute_flux(g, hr, ur)
    fh = (sr*fhl-sl*fhr+sr*sl*(hr-hl))/(sr-sl)
    fu = (sr*ful-sl*fur-sr*sl*ulr)/(sr-sl)

    return (fh, fu)

@njit(cache=True)
def _compute_flux(g, h, u):
    """Auxiliary function for the model integration."""
    fh = h*u
    fu = h*u**2 + 0.5*g*h**2
    return (fh, fu)

