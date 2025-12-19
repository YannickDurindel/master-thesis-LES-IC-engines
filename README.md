# Master Thesis: Large Eddy Simulation of Internal Combustion Engines

Master's thesis on wall-modeled Large Eddy Simulation (LES) of in-cylinder flows using the enhanced viscosity wall treatment approach.

## Overview

This thesis investigates the application of an enhanced viscosity wall treatment for LES of internal combustion engine flows. The work is conducted at the **University of Duisburg-Essen** using the **PsiPhi** CFD solver with immersed boundary methods.

### Key Topics

- **Enhanced viscosity wall treatment** for coarse-grid LES
- **Immersed boundary method** for complex moving geometries
- **Channel flow validation** at Re_τ = 395 against DNS
- **Optical engine simulation** (AVL 5811 single-cylinder research engine)

## Repository Structure

```
├── chapters/           # LaTeX source files for each chapter
├── figures/            # Generated figures and schematics
├── scripts/            # Python post-processing scripts
│   ├── plot_cm_comparison.py    # Cm coefficient comparison
│   ├── plot_dns_fields.py       # DNS/LES visualization
│   └── schematics.py            # Schematic figure generation
├── main.tex            # Main LaTeX document
└── main.pdf            # Compiled thesis
```

## Optical Engine Specifications

| Parameter | Value |
|-----------|-------|
| Engine type | AVL 5811 single-cylinder |
| Bore | 84 mm |
| Stroke | 90 mm |
| Displacement | 499 cm³ |
| Compression ratio | ~10:1 |
| Valve configuration | 4-valve pentroof |

## Building the Thesis

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Author

**Yannick Durindel**
MSc Mechanical Engineering
University of Duisburg-Essen, Germany
