"""
The :py:class:`ASEParser` class is a parser for ASE atoms object.
It is used to extract the necessary data for diffusion analysis from ASE.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Josh Dunn (jd15489).

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from tqdm import tqdm

from kinisi.parser import Parser


class ASEParser(Parser):
    """
    A parser for ASE Atoms objects

    :param atoms: Atoms ordered in sequence of run.
    :param specie: symbol to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling frequency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, defaults
        to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed.
    """

    def __init__(
        self,
        atoms: list['ase.atoms.Atoms'],
        specie: str,
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
        dt: VariableLikeType = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: VariableLikeType = None,
        drift_indices: VariableLikeType = None,
        masses: VariableLikeType = None,
        progress: bool = True,
    ):
        atoms, coords, latt = self.get_structure_coords_latt(atoms, distance_unit, progress)

        if specie is None and specie_indices is None:
            raise TypeError('Must specify specie or specie_indices as scipp VariableLikeType')
        else:
            if specie is not None:
                specie_indices, drift_indices = self.get_indices(atoms, specie)
        super().__init__(
            coords=coords,
            latt=latt,
            time_step=time_step,
            step_skip=step_skip,
            dt=dt,
            specie_indices=specie_indices,
            drift_indices=drift_indices,
            masses=masses,
            dimension=dimension,
        )

    def get_structure_coords_latt(
        self, atoms: list['ase.atoms.Atoms'], distance_unit: sc.Unit, progress: bool = True
    ) -> tuple['ase.atoms.Atoms', VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure and displacement from a :py:attr:`list` of :py:class`pymatgen.core.structure.Structure`.

        :param structures: Structures ordered in sequence of run.
        :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, default is :py:attr:`1`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :return: Tuple containing: initial structure, fractional coordinates for all atoms, and lattice descriptions.
        """
        coords, latt = [], []
        first = True
        if progress:
            iterator = tqdm(atoms, desc='Reading Trajectory')
        else:
            iterator = atoms
        for struct in iterator:
            if first:
                structure = struct
                first = False
            scaled_positions = struct.get_scaled_positions()
            coords.append(np.array(scaled_positions))
            latt.append(struct.cell[:])

        coords.insert(0, coords[0])
        latt.insert(0, latt[0])

        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords, unit=sc.units.dimensionless)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: 'ase.atoms.Atoms',
        specie: str,
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Determine framework and non-framework indices for a :py:mod:`ase` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.

        :returns: Tuple containing: indices for the atoms in the trajectory used in the calculation of the diffusion
            and indices of framework atoms.
        """
        indices = []
        drift_indices = []
        if not isinstance(specie, list):
            specie = [specie]
        for i, site in enumerate(structure):
            if site.symbol in specie:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError('There are no species selected to calculate the mean-squared displacement of.')

        indices = sc.Variable(dims=['atom'], values=indices)
        drift_indices = sc.Variable(dims=['atom'], values=drift_indices)

        return indices, drift_indices
