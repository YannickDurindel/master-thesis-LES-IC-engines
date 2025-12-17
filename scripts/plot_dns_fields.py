#!/usr/bin/env python3
"""
DNS Channel Flow Visualization
==============================
Creates publication-quality visualizations of DNS velocity fields
for the thesis.
"""

import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

# Data directories
COMPARISON_DIR = '/home/yannick/Documents/Education/Deuschland/msr/research/Comparison'
DNS_DIR = os.path.join(COMPARISON_DIR, 'DNS')
LES_DIR = os.path.join(COMPARISON_DIR, '4.0', 'LES')

# Simulation parameters
DELTA = 0.06  # Half channel height [m]
U_B = 1.0     # Bulk velocity [m/s]


def load_velocity_field(data_dir, timestep=None):
    """Load a single velocity field from HDF5 file."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, 'U____.Kmid.*.h5')))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    # Use specified timestep or last one
    if timestep is None:
        filepath = h5_files[-1]
    else:
        filepath = h5_files[min(timestep, len(h5_files)-1)]

    with h5py.File(filepath, 'r') as f:
        data = f['U____'][:]

    return data


def load_averaged_velocity(data_dir):
    """Load and time-average velocity from all HDF5 files."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, 'U____.Kmid.*.h5')))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    print(f"Averaging {len(h5_files)} files from {os.path.basename(data_dir)}")

    # Read first file to get shape
    with h5py.File(h5_files[0], 'r') as f:
        shape = f['U____'].shape

    velocity_sum = np.zeros(shape, dtype=np.float64)
    count = 0

    for filepath in h5_files:
        with h5py.File(filepath, 'r') as f:
            velocity_sum += f['U____'][:]
            count += 1

    return velocity_sum / count


def create_instantaneous_field_plot():
    """Create instantaneous velocity field visualization."""
    print("Creating instantaneous velocity field plot...")

    # Load DNS field
    dns_field = load_velocity_field(DNS_DIR)
    ny, nx = dns_field.shape

    # Create coordinate arrays
    x = np.linspace(0, 2*np.pi*DELTA, nx)
    y = np.linspace(0, 2*DELTA, ny)
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot velocity field
    vmin, vmax = 0, 1.6
    im = ax.pcolormesh(X*1000, Y*1000, dns_field,
                       cmap='RdYlBu_r',
                       shading='auto',
                       vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Streamwise velocity [m/s]',
                        shrink=0.8, aspect=20)

    # Labels
    ax.set_xlabel('Streamwise position [mm]')
    ax.set_ylabel('Wall-normal position [mm]')
    ax.set_title('Instantaneous DNS Velocity Field ($Re_\\tau = 395$)')

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save
    output_path = os.path.join(FIGURES_DIR, 'dns_instantaneous_velocity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_averaged_field_plot():
    """Create time-averaged velocity field visualization."""
    print("Creating time-averaged velocity field plot...")

    # Load averaged DNS field
    dns_avg = load_averaged_velocity(DNS_DIR)
    ny, nx = dns_avg.shape

    # Create coordinate arrays
    x = np.linspace(0, 2*np.pi*DELTA, nx)
    y = np.linspace(0, 2*DELTA, ny)
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot velocity field
    vmin, vmax = 0, 1.6
    im = ax.pcolormesh(X*1000, Y*1000, dns_avg,
                       cmap='viridis',
                       shading='auto',
                       vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Mean streamwise velocity [m/s]',
                        shrink=0.8, aspect=20)

    # Labels
    ax.set_xlabel('Streamwise position [mm]')
    ax.set_ylabel('Wall-normal position [mm]')
    ax.set_title('Time-Averaged DNS Velocity Field ($Re_\\tau = 395$, 501 samples)')

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save
    output_path = os.path.join(FIGURES_DIR, 'dns_averaged_velocity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_dns_les_comparison_contour():
    """Create side-by-side DNS vs LES contour comparison."""
    print("Creating DNS vs LES contour comparison...")

    # Load averaged fields
    dns_avg = load_averaged_velocity(DNS_DIR)
    les_avg = load_averaged_velocity(LES_DIR)

    dns_ny, dns_nx = dns_avg.shape
    les_ny, les_nx = les_avg.shape

    # Create coordinate arrays
    dns_x = np.linspace(0, 2*np.pi*DELTA, dns_nx)
    dns_y = np.linspace(0, 2*DELTA, dns_ny)
    DNS_X, DNS_Y = np.meshgrid(dns_x, dns_y)

    les_x = np.linspace(0, 2*np.pi*DELTA, les_nx)
    les_y = np.linspace(0, 2*DELTA, les_ny)
    LES_X, LES_Y = np.meshgrid(les_x, les_y)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    vmin, vmax = 0, 1.6

    # DNS plot
    im1 = ax1.pcolormesh(DNS_X*1000, DNS_Y*1000, dns_avg,
                         cmap='viridis', shading='auto',
                         vmin=vmin, vmax=vmax)
    ax1.set_ylabel('y [mm]')
    ax1.set_title('DNS (Reference)')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, label='U [m/s]', shrink=0.6)

    # LES plot
    im2 = ax2.pcolormesh(LES_X*1000, LES_Y*1000, les_avg,
                         cmap='viridis', shading='auto',
                         vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Streamwise position [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title('LES with Enhanced Viscosity ($C_m = 4.0$)')
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, label='U [m/s]', shrink=0.6)

    plt.tight_layout()

    # Save
    output_path = os.path.join(FIGURES_DIR, 'dns_les_contour_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_velocity_profile_comparison():
    """Create DNS vs LES velocity profile comparison with wall units."""
    print("Creating velocity profile comparison...")

    # Load averaged fields
    dns_avg = load_averaged_velocity(DNS_DIR)
    les_avg = load_averaged_velocity(LES_DIR)

    # Compute spatially-averaged profiles
    dns_profile = np.mean(dns_avg, axis=1)  # Average along x
    les_profile = np.mean(les_avg, axis=1)

    # Skip ghost cells
    dns_profile = dns_profile[2:-2]
    les_profile = les_profile[3:-3]

    dns_ny = len(dns_profile)
    les_ny = len(les_profile)

    # Create y coordinates
    dns_y = np.linspace(0, 2*DELTA, dns_ny)
    les_y = np.linspace(0, 2*DELTA, les_ny)

    # Average symmetric halves (bottom half)
    dns_half = dns_ny // 2
    les_half = les_ny // 2

    dns_profile_half = (dns_profile[:dns_half] + dns_profile[dns_half:][::-1][:dns_half]) / 2
    les_profile_half = (les_profile[:les_half] + les_profile[les_half:][::-1][:les_half]) / 2

    dns_y_half = dns_y[:dns_half]
    les_y_half = les_y[:les_half]

    # Add wall point
    dns_y_half = np.concatenate([[0], dns_y_half])
    les_y_half = np.concatenate([[0], les_y_half])
    dns_profile_half = np.concatenate([[0], dns_profile_half])
    les_profile_half = np.concatenate([[0], les_profile_half])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Physical units
    ax1.plot(dns_y_half*1000, dns_profile_half, 'b-', linewidth=2, label='DNS')
    ax1.plot(les_y_half*1000, les_profile_half, 'r--', linewidth=2, label='LES ($C_m = 4.0$)')
    ax1.set_xlabel('Distance from wall [mm]')
    ax1.set_ylabel('Mean velocity [m/s]')
    ax1.set_title('Velocity Profile (Physical Units)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, DELTA*1000])
    ax1.set_ylim([0, 1.6])

    # Wall units (log scale)
    # Compute friction velocity from wall gradient
    nu = 1.38e-5  # kinematic viscosity
    u_tau = 0.0909  # From Re_tau = 395

    dns_yplus = dns_y_half * u_tau / nu
    les_yplus = les_y_half * u_tau / nu
    dns_uplus = dns_profile_half / u_tau
    les_uplus = les_profile_half / u_tau

    # Law of wall
    yplus_log = np.logspace(0, 2.5, 100)
    uplus_viscous = yplus_log  # u+ = y+ in viscous sublayer
    uplus_log = 2.44 * np.log(yplus_log) + 5.2  # Log law

    ax2.semilogx(dns_yplus[1:], dns_uplus[1:], 'b-', linewidth=2, label='DNS')
    ax2.semilogx(les_yplus[1:], les_uplus[1:], 'r--', linewidth=2, label='LES ($C_m = 4.0$)')
    ax2.semilogx(yplus_log[yplus_log < 10], uplus_viscous[yplus_log < 10],
                 'k:', linewidth=1.5, label='$u^+ = y^+$')
    ax2.semilogx(yplus_log[yplus_log > 20], uplus_log[yplus_log > 20],
                 'k-.', linewidth=1.5, label='Log law')
    ax2.set_xlabel('$y^+$')
    ax2.set_ylabel('$u^+$')
    ax2.set_title('Velocity Profile (Wall Units)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 400])
    ax2.set_ylim([0, 25])

    plt.tight_layout()

    # Save
    output_path = os.path.join(FIGURES_DIR, 'dns_les_velocity_profiles.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualization figures."""
    print("=" * 60)
    print("DNS/LES Channel Flow Visualization")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    create_instantaneous_field_plot()
    create_averaged_field_plot()
    create_dns_les_comparison_contour()
    create_velocity_profile_comparison()

    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
