"""
Schematic Diagrams for Thesis
=============================
Generate conceptual diagrams using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_config import setup_thesis_style, COLORS, save_figure, get_figure_path

setup_thesis_style()


def create_energy_cascade():
    """
    Create energy cascade diagram showing large to small eddies.
    Figure 2.3 in thesis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background gradient (energy level)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 4, 50)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X/12  # Energy decreases left to right

    ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.3)

    # Draw eddies of decreasing size
    eddy_positions = [
        (1.5, 2, 1.2, COLORS['red']),      # Large energy-containing
        (3.5, 2, 0.8, COLORS['orange']),   # Medium
        (5.0, 2, 0.5, COLORS['orange']),   # Medium-small
        (6.2, 2, 0.35, COLORS['teal']),    # Small
        (7.2, 2, 0.25, COLORS['teal']),    # Smaller
        (8.0, 2, 0.15, COLORS['blue']),    # Tiny
        (8.5, 2, 0.10, COLORS['blue']),    # Tinier
        (9.0, 2, 0.06, COLORS['purple']),  # Smallest
    ]

    for x_pos, y_pos, size, color in eddy_positions:
        # Draw spiral to represent eddy
        theta = np.linspace(0, 4*np.pi, 100)
        r = size * (1 - theta/(8*np.pi))
        x_spiral = x_pos + r * np.cos(theta)
        y_spiral = y_pos + r * np.sin(theta)
        ax.plot(x_spiral, y_spiral, color=color, linewidth=2, alpha=0.8)

    # Energy transfer arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2,
                      connectionstyle='arc3,rad=0.1')
    for i in range(len(eddy_positions)-1):
        x1, y1 = eddy_positions[i][0] + eddy_positions[i][2], eddy_positions[i][1] + 0.5
        x2, y2 = eddy_positions[i+1][0] - eddy_positions[i+1][2], eddy_positions[i+1][1] + 0.5
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Labels and regions
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=6.5, color='gray', linestyle='--', alpha=0.5)

    ax.text(1.5, 3.7, 'Energy-containing\nrange', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 3.7, 'Inertial\nsubrange', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.0, 3.7, 'Dissipation\nrange', ha='center', fontsize=11, fontweight='bold')

    # Energy input and dissipation symbols
    ax.annotate('Energy\ninput', xy=(0.5, 2), xytext=(0.3, 3.2),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))

    ax.annotate('Heat\n(dissipation)', xy=(9.5, 2), xytext=(9.5, 0.5),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2))

    # Scale labels
    ax.text(1.5, 0.3, r'Large scales ($L$)', ha='center', fontsize=10)
    ax.text(8.0, 0.3, r'Kolmogorov scale ($\eta$)', ha='center', fontsize=10)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Energy Cascade: From Large Eddies to Dissipation', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'energy_cascade')
    plt.close()


def create_reynolds_decomposition():
    """
    Create Reynolds decomposition visualization: U = Ū + u'
    Figure 2.4 in thesis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    t = np.linspace(0, 10, 500)

    # Generate signals
    mean_velocity = 5.0 * np.ones_like(t)
    fluctuation = 0.8 * np.sin(2*np.pi*0.5*t) + 0.5 * np.sin(2*np.pi*1.3*t) + \
                  0.3 * np.sin(2*np.pi*3.1*t) + 0.4 * np.random.randn(len(t))
    instantaneous = mean_velocity + fluctuation

    # Panel 1: Instantaneous velocity
    axes[0].plot(t, instantaneous, color=COLORS['blue'], linewidth=1.5)
    axes[0].axhline(y=5.0, color=COLORS['red'], linestyle='--', linewidth=2, label=r'$\bar{U}$')
    axes[0].fill_between(t, mean_velocity, instantaneous, alpha=0.3, color=COLORS['blue'])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title(r'Instantaneous: $U(t)$', fontsize=12, fontweight='bold')
    axes[0].set_ylim(2, 8)
    axes[0].legend(loc='upper right')

    # Panel 2: Mean velocity
    axes[1].axhline(y=5.0, color=COLORS['red'], linewidth=3)
    axes[1].fill_between(t, 0, mean_velocity, alpha=0.3, color=COLORS['red'])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title(r'Mean: $\bar{U}$', fontsize=12, fontweight='bold')
    axes[1].set_ylim(2, 8)

    # Panel 3: Fluctuation
    axes[2].plot(t, fluctuation, color=COLORS['teal'], linewidth=1.5)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].fill_between(t, 0, fluctuation, where=(fluctuation > 0), alpha=0.3, color=COLORS['teal'])
    axes[2].fill_between(t, 0, fluctuation, where=(fluctuation < 0), alpha=0.3, color=COLORS['orange'])
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Velocity')
    axes[2].set_title(r"Fluctuation: $u'(t)$", fontsize=12, fontweight='bold')
    axes[2].set_ylim(-3, 3)

    # Add equation between panels
    fig.text(0.35, 0.02, '=', fontsize=24, fontweight='bold', ha='center')
    fig.text(0.67, 0.02, '+', fontsize=24, fontweight='bold', ha='center')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    # Add main equation
    fig.text(0.5, 0.95, r"Reynolds Decomposition: $U = \bar{U} + u'$",
             fontsize=14, fontweight='bold', ha='center')

    save_figure(fig, 'reynolds_decomposition')
    plt.close()


def create_channel_flow_schematic():
    """
    Create channel flow geometry schematic.
    Figure 2.9 in thesis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Channel walls
    wall_y = 2.0
    ax.fill_between([0, 10], [wall_y, wall_y], [wall_y + 0.3, wall_y + 0.3],
                   color='gray', alpha=0.8, label='Wall')
    ax.fill_between([0, 10], [-wall_y, -wall_y], [-wall_y - 0.3, -wall_y - 0.3],
                   color='gray', alpha=0.8)

    # Velocity profile (parabolic for laminar, log-like for turbulent)
    y = np.linspace(-wall_y, wall_y, 100)
    # Turbulent-like profile
    u = 1.5 * (1 - (np.abs(y)/wall_y)**0.15)

    # Draw velocity profile
    ax.plot(2 + u*1.5, y, color=COLORS['blue'], linewidth=3)
    ax.fill_betweenx(y, 2, 2 + u*1.5, alpha=0.3, color=COLORS['blue'])

    # Velocity arrows
    for yi in np.linspace(-1.5, 1.5, 7):
        ui = 1.5 * (1 - (np.abs(yi)/wall_y)**0.15)
        ax.annotate('', xy=(2 + ui*1.5, yi), xytext=(2, yi),
                   arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1.5))

    # Flow direction arrow
    ax.annotate('', xy=(9, 0), xytext=(6, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=3))
    ax.text(7.5, 0.3, 'Flow direction', fontsize=11, ha='center')

    # Periodic BC arrows
    ax.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
               arrowprops=dict(arrowstyle='<->', color=COLORS['teal'], lw=2))
    ax.text(0, -0.5, 'Periodic', fontsize=10, ha='center', color=COLORS['teal'])

    ax.annotate('', xy=(10.3, 0), xytext=(9.7, 0),
               arrowprops=dict(arrowstyle='<->', color=COLORS['teal'], lw=2))
    ax.text(10, -0.5, 'Periodic', fontsize=10, ha='center', color=COLORS['teal'])

    # Dimensions
    ax.annotate('', xy=(1, wall_y), xytext=(1, -wall_y),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.6, 0, r'$2\delta$', fontsize=12, ha='center', va='center')

    ax.annotate('', xy=(0, -wall_y-0.5), xytext=(10, -wall_y-0.5),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(5, -wall_y-0.8, r'$L_x$', fontsize=12, ha='center')

    # Labels
    ax.text(2, wall_y + 0.5, 'Upper wall (no-slip)', fontsize=10, ha='left')
    ax.text(2, -wall_y - 0.5, 'Lower wall (no-slip)', fontsize=10, ha='left')
    ax.text(3.5, 1.0, r'$U(y)$', fontsize=12, color=COLORS['blue'])

    # Coordinate system
    ax.annotate('', xy=(9, -1.5), xytext=(8.5, -1.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8.5, -1), xytext=(8.5, -1.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(9.1, -1.5, 'x', fontsize=11)
    ax.text(8.5, -0.8, 'y', fontsize=11)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Channel Flow Configuration', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure(fig, 'channel_flow_schematic')
    plt.close()


def create_wall_layer_structure():
    """
    Create near-wall boundary layer structure diagram.
    Figure 2.7 in thesis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Wall
    ax.fill_between([0, 8], [0, 0], [-0.5, -0.5], color='gray', alpha=0.9)
    ax.text(4, -0.25, 'WALL', fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Layer regions (y+ boundaries)
    # Viscous sublayer: y+ < 5
    # Buffer layer: 5 < y+ < 30
    # Log layer: y+ > 30

    y_viscous = 0.5
    y_buffer = 1.5
    y_log = 4.0

    # Viscous sublayer
    ax.fill_between([0, 8], [0, 0], [y_viscous, y_viscous],
                   color=COLORS['blue'], alpha=0.3)
    ax.text(6.5, y_viscous/2, 'Viscous sublayer\n($y^+ < 5$)',
           fontsize=10, ha='center', va='center')

    # Buffer layer
    ax.fill_between([0, 8], [y_viscous, y_viscous], [y_buffer, y_buffer],
                   color=COLORS['orange'], alpha=0.3)
    ax.text(6.5, (y_viscous + y_buffer)/2, 'Buffer layer\n($5 < y^+ < 30$)',
           fontsize=10, ha='center', va='center')

    # Log layer
    ax.fill_between([0, 8], [y_buffer, y_buffer], [y_log, y_log],
                   color=COLORS['teal'], alpha=0.3)
    ax.text(6.5, (y_buffer + y_log)/2, 'Logarithmic layer\n($y^+ > 30$)',
           fontsize=10, ha='center', va='center')

    # Outer layer
    ax.fill_between([0, 8], [y_log, y_log], [5.5, 5.5],
                   color=COLORS['purple'], alpha=0.2)
    ax.text(6.5, (y_log + 5.5)/2, 'Outer layer\n(wake region)',
           fontsize=10, ha='center', va='center')

    # Velocity profile
    y = np.linspace(0.01, 5.5, 200)
    # Composite profile approximation
    u = np.zeros_like(y)
    for i, yi in enumerate(y):
        if yi < 0.5:  # Viscous (y+ < 5 approx)
            u[i] = yi * 8  # Linear
        elif yi < 1.5:  # Buffer
            u[i] = 4 + (yi - 0.5) * 3
        else:  # Log
            u[i] = 5.5 + 2.5 * np.log(yi/1.5 + 1)

    ax.plot(u/2 + 0.5, y, color=COLORS['red'], linewidth=3, label='Velocity profile')

    # Physics descriptions
    ax.text(0.3, 0.25, 'Viscous forces\ndominate', fontsize=9, style='italic')
    ax.text(0.3, 1.0, 'Transition\nregion', fontsize=9, style='italic')
    ax.text(0.3, 2.5, 'Turbulent mixing\ndominates', fontsize=9, style='italic')

    # Dashed lines for boundaries
    ax.axhline(y=y_viscous, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=y_buffer, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=y_log, color='black', linestyle='--', alpha=0.5)

    # y+ labels on right
    ax.text(8.2, 0, '$y^+ = 0$', fontsize=10)
    ax.text(8.2, y_viscous, '$y^+ = 5$', fontsize=10)
    ax.text(8.2, y_buffer, '$y^+ = 30$', fontsize=10)

    ax.set_xlim(0, 9)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')

    ax.set_title('Near-Wall Boundary Layer Structure', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure(fig, 'wall_layer_structure')
    plt.close()


def create_dns_rans_les_comparison():
    """
    Create DNS/RANS/LES scale comparison diagram.
    Figure 2.5 in thesis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Generate turbulent-like field
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 2*np.pi, n)
    y = np.linspace(0, 2*np.pi, n)
    X, Y = np.meshgrid(x, y)

    # Multi-scale field
    field = (np.sin(X) * np.cos(Y) +
            0.5 * np.sin(3*X) * np.cos(3*Y) +
            0.25 * np.sin(7*X) * np.cos(7*Y) +
            0.1 * np.sin(15*X) * np.cos(15*Y) +
            0.05 * np.random.randn(n, n))

    # DNS - full resolution
    axes[0].contourf(X, Y, field, levels=30, cmap='RdBu_r')
    axes[0].set_title('DNS\n(All scales resolved)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Coarse grid overlay
    for i in range(0, n, 5):
        axes[0].axhline(y=y[i], color='black', alpha=0.1, linewidth=0.5)
        axes[0].axvline(x=x[i], color='black', alpha=0.1, linewidth=0.5)

    # LES - filtered (large + some medium scales)
    from scipy.ndimage import gaussian_filter
    field_les = gaussian_filter(field, sigma=3)
    axes[1].contourf(X, Y, field_les, levels=30, cmap='RdBu_r')
    axes[1].set_title('LES\n(Large scales resolved,\nsmall scales modeled)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')

    # Coarser grid
    for i in range(0, n, 10):
        axes[1].axhline(y=y[i], color='black', alpha=0.2, linewidth=0.5)
        axes[1].axvline(x=x[i], color='black', alpha=0.2, linewidth=0.5)

    # RANS - only mean (highly smoothed)
    field_rans = gaussian_filter(field, sigma=15)
    axes[2].contourf(X, Y, field_rans, levels=30, cmap='RdBu_r')
    axes[2].set_title('RANS\n(Only mean flow,\nall turbulence modeled)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')

    # Coarsest grid
    for i in range(0, n, 20):
        axes[2].axhline(y=y[i], color='black', alpha=0.3, linewidth=0.5)
        axes[2].axvline(x=x[i], color='black', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_figure(fig, 'dns_rans_les_comparison')
    plt.close()


def create_four_stroke_cycle():
    """
    Create four-stroke engine cycle diagram.
    Figure 2.11 in thesis.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    strokes = ['Intake', 'Compression', 'Power', 'Exhaust']
    piston_positions = [0.7, 0.3, 0.3, 0.7]
    valve_states = [
        ('open', 'closed'),   # Intake: intake open
        ('closed', 'closed'), # Compression: both closed
        ('closed', 'closed'), # Power: both closed
        ('closed', 'open'),   # Exhaust: exhaust open
    ]

    for ax, stroke, piston_y, (intake_v, exhaust_v) in zip(axes, strokes, piston_positions, valve_states):
        # Cylinder walls
        ax.plot([0.2, 0.2], [0, 1], 'k-', linewidth=3)
        ax.plot([0.8, 0.8], [0, 1], 'k-', linewidth=3)
        ax.plot([0.2, 0.35], [1, 1], 'k-', linewidth=3)
        ax.plot([0.65, 0.8], [1, 1], 'k-', linewidth=3)

        # Cylinder head
        ax.fill_between([0.1, 0.9], [1, 1], [1.1, 1.1], color='gray', alpha=0.5)

        # Piston
        ax.fill_between([0.22, 0.78], [piston_y-0.05, piston_y-0.05],
                       [piston_y+0.05, piston_y+0.05], color='gray', alpha=0.8)
        ax.plot([0.22, 0.78], [piston_y+0.05, piston_y+0.05], 'k-', linewidth=2)

        # Connecting rod
        ax.plot([0.5, 0.5], [piston_y-0.05, piston_y-0.25], 'k-', linewidth=3)

        # Intake valve (left)
        valve_y = 0.95 if intake_v == 'closed' else 0.85
        color = COLORS['blue'] if intake_v == 'open' else 'gray'
        ax.fill([0.35, 0.45, 0.45, 0.35], [valve_y, valve_y, valve_y+0.08, valve_y+0.08],
               color=color, alpha=0.8)

        # Exhaust valve (right)
        valve_y = 0.95 if exhaust_v == 'closed' else 0.85
        color = COLORS['red'] if exhaust_v == 'open' else 'gray'
        ax.fill([0.55, 0.65, 0.65, 0.55], [valve_y, valve_y, valve_y+0.08, valve_y+0.08],
               color=color, alpha=0.8)

        # Flow arrows
        if stroke == 'Intake':
            ax.annotate('', xy=(0.4, 0.7), xytext=(0.4, 1.0),
                       arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
            ax.text(0.25, 0.85, 'Air/fuel\nmixture', fontsize=9, color=COLORS['blue'])
        elif stroke == 'Compression':
            # Swirl arrows
            ax.annotate('', xy=(0.35, 0.6), xytext=(0.5, 0.5),
                       arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5,
                                      connectionstyle='arc3,rad=0.3'))
            ax.annotate('', xy=(0.65, 0.6), xytext=(0.5, 0.5),
                       arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5,
                                      connectionstyle='arc3,rad=-0.3'))
        elif stroke == 'Power':
            # Expansion
            ax.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.7),
                       arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))
            ax.text(0.55, 0.6, 'Combustion', fontsize=9, color=COLORS['red'])
        elif stroke == 'Exhaust':
            ax.annotate('', xy=(0.6, 1.0), xytext=(0.6, 0.7),
                       arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))
            ax.text(0.65, 0.85, 'Exhaust\ngases', fontsize=9, color=COLORS['red'])

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{stroke} Stroke', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'four_stroke_cycle')
    plt.close()


def create_immersed_boundary_schematic():
    """
    Create immersed boundary method schematic.
    Figure 3.6 in thesis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw Cartesian grid
    nx, ny = 12, 10
    for i in range(nx + 1):
        ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(ny + 1):
        ax.axhline(y=j, color='gray', linewidth=0.5, alpha=0.5)

    # Curved surface (immersed boundary)
    x_curve = np.linspace(1, 11, 100)
    y_curve = 3 + 2 * np.sin((x_curve - 1) * np.pi / 10) + 0.5 * np.sin((x_curve - 1) * np.pi / 3)
    ax.plot(x_curve, y_curve, color='black', linewidth=3, label='Immersed boundary')
    ax.fill_between(x_curve, 0, y_curve, color='gray', alpha=0.4)

    # Classify cells
    for i in range(nx):
        for j in range(ny):
            cx, cy = i + 0.5, j + 0.5

            # Check if cell center is below curve
            y_at_cx = np.interp(cx, x_curve, y_curve)

            if cy < y_at_cx - 0.5:
                # Solid cell
                rect = patches.Rectangle((i, j), 1, 1, facecolor=COLORS['gray'], alpha=0.3)
                ax.add_patch(rect)
            elif cy > y_at_cx + 0.5:
                # Fluid cell
                rect = patches.Rectangle((i, j), 1, 1, facecolor=COLORS['blue'], alpha=0.2)
                ax.add_patch(rect)
            elif abs(cy - y_at_cx) <= 0.6:
                # Cut cell / interface cell
                rect = patches.Rectangle((i, j), 1, 1, facecolor=COLORS['orange'], alpha=0.4)
                ax.add_patch(rect)

    # Legend patches
    ax.add_patch(patches.Rectangle((0.5, 9), 0.8, 0.8, facecolor=COLORS['blue'], alpha=0.3))
    ax.text(1.5, 9.4, 'Fluid cells', fontsize=10)

    ax.add_patch(patches.Rectangle((4, 9), 0.8, 0.8, facecolor=COLORS['orange'], alpha=0.5))
    ax.text(5, 9.4, 'Cut cells (IBM)', fontsize=10)

    ax.add_patch(patches.Rectangle((8, 9), 0.8, 0.8, facecolor=COLORS['gray'], alpha=0.4))
    ax.text(9, 9.4, 'Solid cells', fontsize=10)

    # Annotations
    ax.annotate('Curved\nsurface', xy=(6, 5), xytext=(8, 7),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.annotate('Cartesian\ngrid', xy=(2, 8), xytext=(0.5, 8.5),
               fontsize=10, ha='left')

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Immersed Boundary Method: Cell Classification', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure(fig, 'immersed_boundary_schematic')
    plt.close()


def create_staggered_grid():
    """
    Create staggered grid arrangement schematic.
    Figure 3.2 in thesis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Grid lines
    for i in range(5):
        ax.axvline(x=i, color='gray', linewidth=1, linestyle='--')
        ax.axhline(y=i, color='gray', linewidth=1, linestyle='--')

    # Highlight one cell
    cell = patches.Rectangle((1, 1), 1, 1, linewidth=2, edgecolor='black',
                             facecolor=COLORS['blue'], alpha=0.1)
    ax.add_patch(cell)

    # Pressure nodes (cell centers)
    for i in range(4):
        for j in range(4):
            ax.plot(i + 0.5, j + 0.5, 'o', markersize=12, color=COLORS['blue'])
            if i == 1 and j == 1:
                ax.text(i + 0.5, j + 0.3, 'P', fontsize=11, ha='center', color=COLORS['blue'])

    # U-velocity nodes (face centers - vertical faces)
    for i in range(5):
        for j in range(4):
            ax.plot(i, j + 0.5, 's', markersize=10, color=COLORS['red'])
            if i == 1 and j == 1:
                ax.text(i - 0.15, j + 0.5, 'u', fontsize=11, ha='right', color=COLORS['red'])

    # V-velocity nodes (face centers - horizontal faces)
    for i in range(4):
        for j in range(5):
            ax.plot(i + 0.5, j, '^', markersize=10, color=COLORS['teal'])
            if i == 1 and j == 1:
                ax.text(i + 0.5, j - 0.2, 'v', fontsize=11, ha='center', color=COLORS['teal'])

    # Legend
    ax.plot([], [], 'o', markersize=12, color=COLORS['blue'], label='Pressure (cell center)')
    ax.plot([], [], 's', markersize=10, color=COLORS['red'], label='u-velocity (vertical face)')
    ax.plot([], [], '^', markersize=10, color=COLORS['teal'], label='v-velocity (horizontal face)')
    ax.legend(loc='upper right', fontsize=10)

    # Dimension annotations
    ax.annotate('', xy=(2, 0.8), xytext=(1, 0.8),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(1.5, 0.65, r'$\Delta x$', fontsize=12, ha='center')

    ax.annotate('', xy=(0.8, 2), xytext=(0.8, 1),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.65, 1.5, r'$\Delta y$', fontsize=12, ha='center')

    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(-0.3, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Staggered Grid Arrangement', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure(fig, 'staggered_grid')
    plt.close()


def create_control_volume_schematic():
    """
    Create control volume with flux arrows.
    Figure 2.1 in thesis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Control volume
    cv = patches.FancyBboxPatch((2, 2), 4, 3, boxstyle="round,pad=0.05",
                                facecolor=COLORS['blue'], alpha=0.2,
                                edgecolor='black', linewidth=2)
    ax.add_patch(cv)

    # Center point
    ax.plot(4, 3.5, 'ko', markersize=8)
    ax.text(4.2, 3.5, r'$\Omega$', fontsize=14)

    # Flux arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['red'], lw=2)

    # East face
    ax.annotate('', xy=(7, 3.5), xytext=(6, 3.5), arrowprops=arrow_style)
    ax.text(7.2, 3.5, r'$F_e$', fontsize=12, va='center')

    # West face
    ax.annotate('', xy=(1, 3.5), xytext=(2, 3.5), arrowprops=arrow_style)
    ax.text(0.5, 3.5, r'$F_w$', fontsize=12, va='center')

    # North face
    ax.annotate('', xy=(4, 6), xytext=(4, 5), arrowprops=arrow_style)
    ax.text(4, 6.3, r'$F_n$', fontsize=12, ha='center')

    # South face
    ax.annotate('', xy=(4, 1), xytext=(4, 2), arrowprops=arrow_style)
    ax.text(4, 0.5, r'$F_s$', fontsize=12, ha='center')

    # Surface integral notation
    ax.text(6.3, 5.2, r'$\oint_S \mathbf{F} \cdot d\mathbf{A}$', fontsize=14)

    # Boundary labels
    ax.text(6.1, 3.5, 'e', fontsize=10, va='center', style='italic')
    ax.text(1.8, 3.5, 'w', fontsize=10, va='center', style='italic')
    ax.text(4, 5.1, 'n', fontsize=10, ha='center', style='italic')
    ax.text(4, 1.8, 's', fontsize=10, ha='center', style='italic')

    # Volume label
    ax.text(4, 2.5, r'$V = \int_\Omega dV$', fontsize=12, ha='center')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Control Volume with Surface Fluxes', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_figure(fig, 'control_volume')
    plt.close()


def create_tumble_swirl_squish():
    """
    Create tumble, swirl, and squish motion diagram.
    Figure 2.10 in thesis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Tumble motion
    ax = axes[0]
    # Cylinder outline (top view showing side)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2
    ax.plot(r*np.cos(theta), r*np.sin(theta), 'k-', linewidth=2)

    # Tumble vortex (rotating around horizontal axis)
    t = np.linspace(0, 1.8*np.pi, 100)
    r_tumble = 1.5
    x_tumble = r_tumble * np.cos(t)
    y_tumble = r_tumble * np.sin(t)
    ax.plot(x_tumble, y_tumble, color=COLORS['blue'], linewidth=2)
    ax.annotate('', xy=(x_tumble[-1], y_tumble[-1]), xytext=(x_tumble[-5], y_tumble[-5]),
               arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))

    ax.annotate('', xy=(0, 1.8), xytext=(0, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))
    ax.text(0.2, 2.5, 'Intake', fontsize=10, color=COLORS['red'])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Tumble\n(rotation about horizontal axis)', fontsize=12, fontweight='bold')

    # Swirl motion
    ax = axes[1]
    ax.plot(r*np.cos(theta), r*np.sin(theta), 'k-', linewidth=2)

    # Swirl spiral (top view)
    t = np.linspace(0, 3*np.pi, 150)
    r_swirl = 1.5 * (1 - t/(6*np.pi))
    x_swirl = r_swirl * np.cos(t)
    y_swirl = r_swirl * np.sin(t)
    ax.plot(x_swirl, y_swirl, color=COLORS['teal'], linewidth=2)
    ax.annotate('', xy=(x_swirl[-1], y_swirl[-1]), xytext=(x_swirl[-5], y_swirl[-5]),
               arrowprops=dict(arrowstyle='->', color=COLORS['teal'], lw=2))

    ax.text(0, 0, 'Top\nview', fontsize=10, ha='center', va='center', style='italic')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Swirl\n(rotation about cylinder axis)', fontsize=12, fontweight='bold')

    # Squish motion
    ax = axes[2]
    # Side view of piston moving up
    ax.fill_between([-2, 2], [-1.5, -1.5], [-1.2, -1.2], color='gray', alpha=0.6)
    ax.plot([-2, -2], [-1.2, 1.5], 'k-', linewidth=2)
    ax.plot([2, 2], [-1.2, 1.5], 'k-', linewidth=2)
    ax.plot([-2, 2], [1.5, 1.5], 'k-', linewidth=2)

    # Piston
    ax.fill_between([-1.8, 1.8], [0, 0], [0.3, 0.3], color='gray', alpha=0.8)

    # Squish arrows (radial inward at top)
    ax.annotate('', xy=(0, 1.2), xytext=(-1.5, 1.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))
    ax.annotate('', xy=(0, 1.2), xytext=(1.5, 1.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))

    # Piston motion arrow
    ax.annotate('', xy=(0, 0.5), xytext=(0, -0.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))
    ax.text(0.3, 0.1, 'Piston\nmotion', fontsize=9, color=COLORS['red'])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Squish\n(radial flow during compression)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'tumble_swirl_squish')
    plt.close()


def generate_all_schematics():
    """Generate all schematic figures."""
    print("Generating schematic figures...")

    create_energy_cascade()
    print("  Created: energy_cascade")

    create_reynolds_decomposition()
    print("  Created: reynolds_decomposition")

    create_channel_flow_schematic()
    print("  Created: channel_flow_schematic")

    create_wall_layer_structure()
    print("  Created: wall_layer_structure")

    create_dns_rans_les_comparison()
    print("  Created: dns_rans_les_comparison")

    create_four_stroke_cycle()
    print("  Created: four_stroke_cycle")

    create_immersed_boundary_schematic()
    print("  Created: immersed_boundary_schematic")

    create_staggered_grid()
    print("  Created: staggered_grid")

    create_control_volume_schematic()
    print("  Created: control_volume")

    create_tumble_swirl_squish()
    print("  Created: tumble_swirl_squish")

    print("\nAll schematic figures generated!")


if __name__ == "__main__":
    generate_all_schematics()
