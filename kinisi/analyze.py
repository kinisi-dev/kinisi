"""
This module contains the API classes for :py:mod:`kinisi`.
It is anticipated that this is where the majority of interaction with the package will occur.
This module includes:

* the :py:class:`kinisi.analyze.DiffusionAnalyzer` class for MSD and diffusion analysis;
* the :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class for MSTD and collective diffusion analysis; and
* the :py:class:`kinisi.analyze.ConductivityAnalyzer` class for MSCD and conductivity analysis.

These are all compatible with VASP Xdatcar output files, pymatgen structures and any MD trajectory that the
:py:mod:`MDAnalysis` package can handle.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from kinisi.diffusion_analyzer import DiffusionAnalyzer
from kinisi.jump_diffusion_analyzer import JumpDiffusionAnalyzer
from kinisi.conductivity_analyzer import ConductivityAnalyzer
