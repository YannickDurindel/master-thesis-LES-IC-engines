"""
HDF5 Data Loader for Channel Flow DNS
======================================
Utilities to load and process DNS data from PsiPhi9 simulations.
"""

import numpy as np
import h5py
import os
from pathlib import Path

# Data directory
DATA_DIR = Path("/home/yannick/Documents/Education/Deuschland/msr/research/PsiPhi9/HDF5/2D")

# Variable descriptions
VARIABLES = {
    'U____': 'Streamwise velocity',
    'V____': 'Wall-normal velocity',
    'W____': 'Spanwise velocity',
    'P____': 'Pressure',
    'Ufl__': 'Filtered streamwise velocity',
    'Vfl__': 'Filtered wall-normal velocity',
    'Wfl__': 'Filtered spanwise velocity',
    'RU___': 'Reynolds stress u\'u\'',
    'RV___': 'Reynolds stress v\'v\'',
    'RW___': 'Reynolds stress w\'w\'',
    'UVrs_': 'Reynolds stress u\'v\'',
    'UWrs_': 'Reynolds stress u\'w\'',
    'VWrs_': 'Reynolds stress v\'w\'',
    'kres_': 'Resolved turbulent kinetic energy',
    'Rs___': 'SGS Reynolds stress magnitude',
    'Rt___': 'Turbulent SGS stress',
    'Ml___': 'Eddy viscosity (Launder-Reece-Rodi)',
    'Ms___': 'Eddy viscosity (Smagorinsky)',
    'Mt___': 'Eddy viscosity (Total)',
    'Fcs__': 'Coherent structure function',
}

# Slice descriptions
SLICES = {
    'Imid': 'I-direction mid-plane (y-z plane)',
    'Jmid': 'J-direction mid-plane (x-z plane)',
    'Kmid': 'K-direction mid-plane (x-y plane)',
    'VOLK': 'Volume slice',
}


def list_available_files(variable=None, slice_type=None, subdir=None):
    """List available HDF5 files matching criteria."""
    search_dir = DATA_DIR
    if subdir:
        search_dir = DATA_DIR / subdir

    files = []
    for f in search_dir.glob("*.h5"):
        name = f.stem
        parts = name.split('.')
        if len(parts) >= 3:
            var, slc, timestep = parts[0], parts[1], parts[2]
            if variable and var != variable:
                continue
            if slice_type and slc != slice_type:
                continue
            files.append({
                'path': f,
                'variable': var,
                'slice': slc,
                'timestep': int(timestep),
            })

    return sorted(files, key=lambda x: x['timestep'])


def load_hdf5(filepath):
    """Load data from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        # Get all datasets
        data = {}
        for key in f.keys():
            data[key] = f[key][:]

        # Get attributes if any
        attrs = dict(f.attrs)

    return data, attrs


def load_field(variable, slice_type, timestep, subdir=None):
    """Load a specific field from HDF5 data."""
    search_dir = DATA_DIR
    if subdir:
        search_dir = DATA_DIR / subdir

    filename = f"{variable}.{slice_type}.{timestep:07d}.h5"
    filepath = search_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data, attrs = load_hdf5(filepath)

    # Return first dataset (usually the main data)
    if len(data) == 1:
        return list(data.values())[0]
    return data


def load_velocity_field(slice_type, timestep, subdir=None):
    """Load all three velocity components."""
    U = load_field('U____', slice_type, timestep, subdir)
    V = load_field('V____', slice_type, timestep, subdir)
    W = load_field('W____', slice_type, timestep, subdir)
    return U, V, W


def load_reynolds_stresses(slice_type, timestep, subdir=None):
    """Load Reynolds stress components."""
    RU = load_field('RU___', slice_type, timestep, subdir)
    RV = load_field('RV___', slice_type, timestep, subdir)
    RW = load_field('RW___', slice_type, timestep, subdir)
    UV = load_field('UVrs_', slice_type, timestep, subdir)
    return RU, RV, RW, UV


def time_average_fields(variable, slice_type, timesteps=None, subdir=None):
    """Compute time-averaged field."""
    files = list_available_files(variable, slice_type, subdir)

    if timesteps is not None:
        files = [f for f in files if f['timestep'] in timesteps]

    if not files:
        raise ValueError("No files found for averaging")

    # Load first to get shape
    first_data = load_field(variable, slice_type, files[0]['timestep'], subdir)
    avg = np.zeros_like(first_data, dtype=np.float64)

    for f in files:
        data = load_field(variable, slice_type, f['timestep'], subdir)
        avg += data

    return avg / len(files)


def compute_velocity_magnitude(U, V, W=None):
    """Compute velocity magnitude from components."""
    if W is None:
        return np.sqrt(U**2 + V**2)
    return np.sqrt(U**2 + V**2 + W**2)


def compute_tke(RU, RV, RW):
    """Compute turbulent kinetic energy from Reynolds stresses."""
    return 0.5 * (RU + RV + RW)


# Wall-unit scaling functions
class WallUnits:
    """Class for wall-unit scaling of channel flow data."""

    def __init__(self, Re_tau, nu, u_tau=None, delta=1.0):
        """
        Initialize wall unit scaling.

        Parameters:
        -----------
        Re_tau : float
            Friction Reynolds number
        nu : float
            Kinematic viscosity
        u_tau : float, optional
            Friction velocity (computed from Re_tau if not given)
        delta : float
            Channel half-height (default 1.0)
        """
        self.Re_tau = Re_tau
        self.nu = nu
        self.delta = delta

        if u_tau is None:
            self.u_tau = Re_tau * nu / delta
        else:
            self.u_tau = u_tau

        # Viscous length scale
        self.delta_nu = nu / self.u_tau

    def y_plus(self, y):
        """Convert y to y+."""
        return y / self.delta_nu

    def u_plus(self, u):
        """Convert u to u+."""
        return u / self.u_tau

    def y_from_plus(self, y_plus):
        """Convert y+ to y."""
        return y_plus * self.delta_nu

    def u_from_plus(self, u_plus):
        """Convert u+ to u."""
        return u_plus * self.u_tau


# Analytical profiles for validation
def law_of_wall_linear(y_plus):
    """Linear sublayer: u+ = y+."""
    return y_plus


def law_of_wall_log(y_plus, kappa=0.41, B=5.2):
    """Log law: u+ = (1/kappa) * ln(y+) + B."""
    return (1.0 / kappa) * np.log(y_plus) + B


def spalding_profile(y_plus, kappa=0.41, B=5.2):
    """
    Spalding's law of the wall (implicit, valid for all y+).
    Returns u+ for given y+.
    """
    from scipy.optimize import fsolve

    def spalding_eq(u_plus, y_plus_val):
        exp_term = np.exp(kappa * u_plus)
        return (u_plus + exp_term - 1
                - (kappa * u_plus)
                - (kappa * u_plus)**2 / 2
                - (kappa * u_plus)**3 / 6
                - np.exp(kappa * B) * (exp_term - 1
                    - kappa * u_plus
                    - (kappa * u_plus)**2 / 2
                    - (kappa * u_plus)**3 / 6)
                - y_plus_val)

    # Vectorized solution
    if np.isscalar(y_plus):
        u_plus_guess = min(y_plus, 25)
        return fsolve(spalding_eq, u_plus_guess, args=(y_plus,))[0]

    u_plus = np.zeros_like(y_plus)
    for i, yp in enumerate(y_plus):
        u_plus_guess = min(yp, 25)
        u_plus[i] = fsolve(spalding_eq, u_plus_guess, args=(yp,))[0]

    return u_plus


if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...")
    print(f"\nData directory: {DATA_DIR}")

    # Check for subdirectories
    subdirs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Subdirectories found: {subdirs}")

    # List some files
    for subdir in subdirs[:2]:
        print(f"\nFiles in {subdir}:")
        files = list_available_files(subdir=subdir)
        for f in files[:5]:
            print(f"  {f['variable']}.{f['slice']}.{f['timestep']}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
