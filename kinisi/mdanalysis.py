"""
The :py:class:`MDAnalysisParser` class is a parser for MDAnalysis universe object.
It is used to extract the necessary data for diffusion analysis from an MDAnalysis universe.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from tqdm import tqdm

from kinisi.parser import Parser


class MDAnalysisParser(Parser):
    """
    Parser for MDAnalysis structures.

    Takes an MDAnalysis.Universe object as an input.

    :param universe: MDanalysis universe object to be parsed
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note,
        that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit
        that is use for the time interval values.
    :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
        output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
        a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
    :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
        ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with
        a step size the same as the smallest interval.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param distance_unit: The unit of distance used in the input structures. Optional, defaults to angstroms.
    :param sub_sample_atoms: Subsample the atoms in the trajectory. Optional, defaults to 1.
    :param sub_sample_traj: Subsample the trajectory. Optional, defaults to 1.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(
        self,
        universe: 'MDAnalysis.core.universe.Universe',
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
        structure, coords, latt = self.get_structure_coords_latt(universe, distance_unit, progress)

        if specie is None and specie_indices is None:
                raise TypeError('Must specify specie or specie_indices as scipp VariableLikeType')
        else:
            if specie is not None:
                specie_indices, drift_indices = self.get_indices(structure, specie)

        super().__init__(
            structure, coords, latt, specie, time_step, step_skip, dt, specie_indices, masses, dimension
        )

    def get_structure_coords_latt(
        self,
        universe: 'MDAnalysis.core.universe.Universe',
        distance_unit: VariableLikeType,
        progress: bool = True,
    ) -> tuple['MDAnalysis.core.universe.Universe', VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, coordinates, and lattice parameters from an MDAnalysis.Universe object.

        :param universe: MDanalysis universe object.
        :param progress: Whether to show a progress bar when reading in the structures.
        :param sub_sample_atoms: Subsample the atoms in the trajectory. Optional, defaults to 1.
        :param sub_sample_traj: Subsample the trajectory. Optional, defaults to 1.

        :returns: A tuple of:  the initial structure (as
            a :py:class:`MDAnalysis.core.universe.Universe.atoms`), coordinates (as
            a :py:mod:`scipp` array with dimensions of `time`, `atom`, and `dimension`),
            and lattice parameters (as a :py:mod:`scipp` array with dimensions `time`,
            `dimension1`, and `dimension2`).
        """
        first = True
        coords_l = []
        latt_l = []
        if progress:
            iterator = tqdm(universe.trajectory, desc='Reading Trajectory')
        else:
            iterator = universe.trajectory

        for struct in iterator:
            if first:
                structure = universe.atoms
                first = False
            matrix = np.array(struct.triclinic_dimensions)
            inv_matrix = np.linalg.inv(matrix)
            coords_l.append(np.dot(universe.atoms.positions, inv_matrix))
            latt_l.append(np.array(matrix))

        coords_l = np.array(coords_l)
        latt_l = np.array(latt_l)

        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords_l, unit=sc.units.dimensionless)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt_l, unit=distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: 'MDAnalysis.universe.Universe.atoms',
        specie: str,
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Determine framework and non-framework indices for an :py:mod:`MDAnalysis` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        :return: Tuple containing indices for the atoms in the trajectory used in the calculation of the
            diffusion and indices of framework atoms.
        """
        indices = []
        drift_indices = []

        if not isinstance(specie, list):
            specie = [specie]

        for i, site in enumerate(structure):
            if site.type in specie:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError('There are no atoms selected to calculate the mean-squared displacement of.')

        indices = sc.Variable(dims=['atom'], values=indices)
        drift_indices = sc.Variable(dims=['atom'], values=drift_indices)

        return indices, drift_indices
