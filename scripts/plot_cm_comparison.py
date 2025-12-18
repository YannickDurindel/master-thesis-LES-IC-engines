#!/usr/bin/env python3
"""
Cm Coefficient Comparison for Channel Flow LES
===============================================
Compares LES results with different enhanced viscosity coefficients (Cm)
against DNS reference data to demonstrate the trade-off between bulk flow
accuracy and near-wall behavior.

This script generates figures for the thesis showing:
1. Full velocity profile comparison (DNS vs Cm=1.0 vs Cm=4.0)
2. Near-wall region zoom to highlight boundary layer accuracy
3. Wall-unit scaling to show log-law behavior

Key finding: Cm=1.0 provides better bulk flow match, but Cm=4.0 gives
more accurate near-wall gradients - critical for boundary layer studies.
"""

import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# =============================================================================
# Simulation Parameters
# =============================================================================
DELTA = 0.06       # Half channel height [m]
U_B = 1.0          # Bulk velocity [m/s]
U_TAU = 0.0909     # Friction velocity [m/s] (from Re_tau = 395)
NU = 1.38e-5       # Kinematic viscosity [m²/s]
MU = 1.78e-5       # Dynamic viscosity [Pa.s]
RE_TAU = 395       # Friction Reynolds number

# Ghost cells to skip
N_GHOST_DNS = 2
N_GHOST_LES = 3

# =============================================================================
# Data Paths
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
COMPARISON_DIR = '/home/yannick/Documents/Education/Deuschland/msr/research/Comparison'

DNS_DIR = os.path.join(COMPARISON_DIR, 'DNS')
LES_CM10_DIR = os.path.join(COMPARISON_DIR, '1.0', 'LES')
LES_CM40_DIR = os.path.join(COMPARISON_DIR, '4.0', 'LES')


def load_averaged_velocity(data_dir):
    """Load and time-average velocity from all HDF5 files."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, 'U____.Kmid.*.h5')))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    print(f"  Loading {len(h5_files)} files from {os.path.basename(os.path.dirname(data_dir))}/{os.path.basename(data_dir)}")

    with h5py.File(h5_files[0], 'r') as f:
        shape = f['U____'].shape

    velocity_sum = np.zeros(shape, dtype=np.float64)

    for filepath in h5_files:
        with h5py.File(filepath, 'r') as f:
            velocity_sum += f['U____'][:]

    return velocity_sum / len(h5_files)


def compute_profile(velocity_field, n_ghost):
    """Compute mean velocity profile, skipping ghost cells."""
    if n_ghost > 0:
        velocity_field = velocity_field[n_ghost:-n_ghost, :]
    return np.mean(velocity_field, axis=1)


def average_symmetric_halves(profile, y):
    """Average bottom and top halves of channel for symmetry."""
    n = len(profile)
    n_half = n // 2

    bottom = profile[:n_half]
    top = profile[n_half:][::-1][:n_half]

    profile_avg = (bottom + top) / 2
    y_half = y[:n_half]

    # Add wall point (no-slip)
    y_half = np.concatenate([[0], y_half])
    profile_avg = np.concatenate([[0], profile_avg])

    return profile_avg, y_half


def create_y_coords(ny):
    """Create y-coordinate array."""
    dy = 2 * DELTA / ny
    return np.linspace(dy/2, 2*DELTA - dy/2, ny)


def main():
    """Generate Cm comparison figures."""
    print("=" * 60)
    print("Cm Coefficient Comparison: Channel Flow LES")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================================================================
    # Load all datasets
    # =========================================================================
    print("\nLoading velocity data...")

    dns_avg = load_averaged_velocity(DNS_DIR)
    les_cm10_avg = load_averaged_velocity(LES_CM10_DIR)
    les_cm40_avg = load_averaged_velocity(LES_CM40_DIR)

    # =========================================================================
    # Compute profiles
    # =========================================================================
    print("\nComputing velocity profiles...")

    dns_profile = compute_profile(dns_avg, N_GHOST_DNS)
    les_cm10_profile = compute_profile(les_cm10_avg, N_GHOST_LES)
    les_cm40_profile = compute_profile(les_cm40_avg, N_GHOST_LES)

    dns_y = create_y_coords(len(dns_profile))
    les_cm10_y = create_y_coords(len(les_cm10_profile))
    les_cm40_y = create_y_coords(len(les_cm40_profile))

    # Symmetry averaging
    dns_u, dns_y = average_symmetric_halves(dns_profile, dns_y)
    les_cm10_u, les_cm10_y = average_symmetric_halves(les_cm10_profile, les_cm10_y)
    les_cm40_u, les_cm40_y = average_symmetric_halves(les_cm40_profile, les_cm40_y)

    # =========================================================================
    # Compute wall shear stress
    # =========================================================================
    # Wall gradient (first interior point)
    dns_dudy_wall = dns_u[1] / dns_y[1]
    les_cm10_dudy_wall = les_cm10_u[1] / les_cm10_y[1]
    les_cm40_dudy_wall = les_cm40_u[1] / les_cm40_y[1]

    dns_tau_wall = MU * dns_dudy_wall
    les_cm10_tau_wall = MU * les_cm10_dudy_wall
    les_cm40_tau_wall = MU * les_cm40_dudy_wall

    print(f"\nWall shear stress:")
    print(f"  DNS:      tau_w = {dns_tau_wall:.6f} Pa")
    print(f"  Cm=1.0:   tau_w = {les_cm10_tau_wall:.6f} Pa (error: {100*(les_cm10_tau_wall-dns_tau_wall)/dns_tau_wall:+.1f}%)")
    print(f"  Cm=4.0:   tau_w = {les_cm40_tau_wall:.6f} Pa (error: {100*(les_cm40_tau_wall-dns_tau_wall)/dns_tau_wall:+.1f}%)")

    # =========================================================================
    # Compute bulk velocity errors
    # =========================================================================
    dns_u_center = np.max(dns_u)
    les_cm10_u_center = np.max(les_cm10_u)
    les_cm40_u_center = np.max(les_cm40_u)

    print(f"\nCenterline velocity:")
    print(f"  DNS:      U_c = {dns_u_center:.4f} m/s")
    print(f"  Cm=1.0:   U_c = {les_cm10_u_center:.4f} m/s (error: {100*(les_cm10_u_center-dns_u_center)/dns_u_center:+.1f}%)")
    print(f"  Cm=4.0:   U_c = {les_cm40_u_center:.4f} m/s (error: {100*(les_cm40_u_center-dns_u_center)/dns_u_center:+.1f}%)")

    # =========================================================================
    # Figure 1: Full Profile Comparison
    # =========================================================================
    print("\nGenerating figures...")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Physical units
    ax1.plot(dns_y*1000, dns_u, 'k-', linewidth=2.5, label='DNS (Reference)')
    ax1.plot(les_cm10_y*1000, les_cm10_u, 'b--', linewidth=2, label='LES ($C_m = 1.0$)')
    ax1.plot(les_cm40_y*1000, les_cm40_u, 'r-.', linewidth=2, label='LES ($C_m = 4.0$)')

    ax1.set_xlabel('Distance from wall [mm]')
    ax1.set_ylabel('Mean velocity [m/s]')
    ax1.set_title('Full Channel Profile')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, DELTA*1000])
    ax1.set_ylim([0, 1.6])

    # Right: Near-wall zoom
    ax2.plot(dns_y*1000, dns_u, 'k-', linewidth=2.5, label='DNS (Reference)')
    ax2.plot(les_cm10_y*1000, les_cm10_u, 'b--', linewidth=2, label='LES ($C_m = 1.0$)')
    ax2.plot(les_cm40_y*1000, les_cm40_u, 'r-.', linewidth=2, label='LES ($C_m = 4.0$)')

    ax2.set_xlabel('Distance from wall [mm]')
    ax2.set_ylabel('Mean velocity [m/s]')
    ax2.set_title('Near-Wall Region (Boundary Layer)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])  # Zoom to ~25% of half-channel
    ax2.set_ylim([0, 1.2])

    # Add annotation about wall gradient
    ax2.annotate('Higher $C_m$ improves\nwall gradient accuracy',
                 xy=(5, 0.4), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, 'cm_comparison_profiles.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # =========================================================================
    # Figure 2: Wall Units (Log Scale)
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(10, 7))

    # Convert to wall units
    dns_yplus = dns_y * U_TAU / NU
    les_cm10_yplus = les_cm10_y * U_TAU / NU
    les_cm40_yplus = les_cm40_y * U_TAU / NU

    dns_uplus = dns_u / U_TAU
    les_cm10_uplus = les_cm10_u / U_TAU
    les_cm40_uplus = les_cm40_u / U_TAU

    # Reference laws
    yplus_ref = np.logspace(0, 2.7, 200)
    uplus_viscous = yplus_ref  # u+ = y+ (viscous sublayer)
    uplus_log = 2.44 * np.log(yplus_ref) + 5.2  # Log law

    ax.semilogx(dns_yplus[1:], dns_uplus[1:], 'k-', linewidth=2.5, label='DNS')
    ax.semilogx(les_cm10_yplus[1:], les_cm10_uplus[1:], 'b--', linewidth=2, label='LES ($C_m = 1.0$)')
    ax.semilogx(les_cm40_yplus[1:], les_cm40_uplus[1:], 'r-.', linewidth=2, label='LES ($C_m = 4.0$)')
    ax.semilogx(yplus_ref[yplus_ref < 8], uplus_viscous[yplus_ref < 8],
                'k:', linewidth=1.5, label='$u^+ = y^+$')
    ax.semilogx(yplus_ref[yplus_ref > 30], uplus_log[yplus_ref > 30],
                'k-.', linewidth=1.5, alpha=0.7, label='Log law')

    ax.set_xlabel('$y^+$')
    ax.set_ylabel('$u^+$')
    ax.set_title('Velocity Profile in Wall Units ($Re_\\tau = 395$)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 500])
    ax.set_ylim([0, 25])

    # Add regions annotation
    ax.axvspan(1, 5, alpha=0.1, color='blue', label='Viscous sublayer')
    ax.axvspan(30, 400, alpha=0.1, color='green', label='Log layer')
    ax.text(2.5, 22, 'Viscous\nsublayer', fontsize=9, ha='center')
    ax.text(100, 22, 'Log layer', fontsize=9, ha='center')

    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, 'cm_comparison_wall_units.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # =========================================================================
    # Figure 3: Error Analysis
    # =========================================================================
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Interpolate LES profiles to DNS y-coordinates for error computation
    les_cm10_interp = np.interp(dns_y, les_cm10_y, les_cm10_u)
    les_cm40_interp = np.interp(dns_y, les_cm40_y, les_cm40_u)

    # Avoid division by zero at wall
    mask = dns_u > 0.01

    error_cm10 = 100 * (les_cm10_interp - dns_u) / np.maximum(dns_u, 0.01)
    error_cm40 = 100 * (les_cm40_interp - dns_u) / np.maximum(dns_u, 0.01)

    # Left: Absolute error
    ax1.plot(dns_y[mask]*1000, les_cm10_interp[mask] - dns_u[mask],
             'b-', linewidth=2, label='$C_m = 1.0$')
    ax1.plot(dns_y[mask]*1000, les_cm40_interp[mask] - dns_u[mask],
             'r--', linewidth=2, label='$C_m = 4.0$')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Distance from wall [mm]')
    ax1.set_ylabel('Velocity error [m/s]')
    ax1.set_title('Absolute Error (LES - DNS)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, DELTA*1000])

    # Right: Near-wall error zoom
    near_wall = dns_y < 0.015  # First 15mm
    ax2.plot(dns_y[near_wall]*1000, (les_cm10_interp - dns_u)[near_wall],
             'b-', linewidth=2, label='$C_m = 1.0$')
    ax2.plot(dns_y[near_wall]*1000, (les_cm40_interp - dns_u)[near_wall],
             'r--', linewidth=2, label='$C_m = 4.0$')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Distance from wall [mm]')
    ax2.set_ylabel('Velocity error [m/s]')
    ax2.set_title('Near-Wall Error (Boundary Layer Region)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])

    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, 'cm_comparison_error.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    print("\nAll figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
