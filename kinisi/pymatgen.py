"""
The `PymatgenParser` class is a parser for pymatgen structures for :py:mod:`kinisi`.
It is used to extract the necessary data for diffusion analysis from a list of pymatgen structures.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

from typing import Union

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from tqdm import tqdm

from kinisi.parser import Parser


class PymatgenParser(Parser):
    """
    Parser for pymatgen structures.

    This takes a list of pymatgen structures as an input.

    :param structures: Structures ordered in sequence of run.
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
    :param specie_indices: Indices of the specie to calculate the diffusivity for. Optional, defaults to `None`.
    :param masses: Masses of the atoms in the structure. Optional, defaults to `None`.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(
        self,
        structures: list['pymatgen.core.structure.Structure'],
        specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
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
        structure, coords, latt = self.get_structure_coords_latt(structures, distance_unit, progress)

        if specie is None and specie_indices is None:
                raise TypeError('Must specify specie or specie_indices as scipp VariableLikeType')
        else:
            if specie is not None:
                specie_indices, drift_indices = self.get_indices(structure, specie)

        super().__init__(
            coords=coords,
            latt=latt,
            time_step=time_step,
            step_skip=step_skip,
            dt=dt,
            specie_indices=specie_indices,
            drift_indices=drift_indices,
            masses=masses,
            dimension=dimension
        )

    def get_structure_coords_latt(
        self,
        structures: list['pymatgen.core.structure.Structure'],
        distance_unit: VariableLikeType,
        progress: bool = True,
    ) -> tuple['pymatgen.core.structure.Structure', VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, coordinates, and lattice parameters from a list of pymatgen structures.

        :param structures: Structures ordered in sequence of run.
        :param progress: Whether to show a progress bar when reading in the structures.

        :returns: A tuple of the initial structure (as
            a :py:class:`pymatgen.core.structure.Structure`), coordinates (as
            a :py:mod:`scipp` array with dimensions of `time`, `atom`, and `dimension`),
            and lattice parameters (as a :py:mod:`scipp` array with dimensions `time`,
            `dimension1`, and `dimension2`).
        """
        first = True
        coords_l = []
        latt_l = []
        if progress:
            iterator = tqdm(structures, desc='Reading Trajectory')
        else:
            iterator = structures

        for struct in iterator:
            if first:
                structure = struct
                first = False
            coords_l.append(np.array(struct.frac_coords))
            latt_l.append(np.array(struct.lattice.matrix))

        coords_l.insert(0, coords_l[0])
        latt_l.insert(0, latt_l[0])
        coords_l = np.array(coords_l)
        latt_l = np.array(latt_l)

        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords_l, unit=sc.units.dimensionless)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt_l, unit=distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: 'pymatgen.core.structure.Structure',
        specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Determine the framework and mobile indices from a :py:mod:`pymatgen` structure.

        :param structure: The initial structure to determine the indices from.
        :param specie: The specie to calculate the diffusivity for.

        :returns: A tuple of the indices for the specie of interest (mobile) and the
            drift (framework) indices.
        """
        indices = []
        drift_indices = []
        for i, site in enumerate(structure):
            if site.specie.__str__() in specie:
                indices.append(i)
            else:
                drift_indices.append(i)
        indices = sc.Variable(dims=['atom'], values=indices)
        drift_indices = sc.Variable(dims=['atom'], values=drift_indices)
        return indices, drift_indices
