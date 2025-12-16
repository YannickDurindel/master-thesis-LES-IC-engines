"""
Thesis Figure Configuration
===========================
Publication-quality matplotlib settings for master thesis figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Publication-quality settings
def setup_thesis_style():
    """Configure matplotlib for thesis figures."""
    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams.update({
        # Figure size (in inches) - good for A4 thesis
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Fonts
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # LaTeX
        'text.usetex': False,  # Set True if LaTeX is available
        'mathtext.fontset': 'cm',
    })

# Colorblind-friendly color palette
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'purple': '#AA4499',
    'gray': '#BBBBBB',
    'cyan': '#33BBEE',
    'magenta': '#EE3377',
}

# Color lists for line plots
LINE_COLORS = [COLORS['blue'], COLORS['orange'], COLORS['teal'],
               COLORS['red'], COLORS['purple']]

# Colormaps for contour plots
CMAPS = {
    'velocity': 'viridis',
    'diverging': 'RdBu_r',
    'pressure': 'coolwarm',
    'turbulence': 'plasma',
    'sequential': 'inferno',
}

def get_figure_path(name, extension='pdf'):
    """Return the standard path for a thesis figure."""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'figures', f'{name}.{extension}')

def save_figure(fig, name, formats=['pdf', 'png']):
    """Save figure in multiple formats for thesis."""
    for fmt in formats:
        path = get_figure_path(name, fmt)
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {path}")

# Initialize style when module is imported
setup_thesis_style()
