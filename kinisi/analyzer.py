"""
The Analyzer class is the main API interface for users to :py:mod:`kinisi`.
Typically, it is expected that the user would access a specific :py:class:`Analyzer`,
such as the :py:class:`DiffusionAnalyzer`, which sub-class this class.

Note, that all of the :py:attr:`classmethod` functions for the :py:class:`Analyzer`
are intended for internal use.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

from typing import Union

import numpy as np
import scipp as sc

from kinisi.ase import ASEParser
from kinisi.mdanalysis import MDAnalysisParser
from kinisi.parser import Parser
from kinisi.pymatgen import PymatgenParser


class Analyzer:
    """
    This class is the superclass for the :py:class:`kinisi.analyze.DiffusionAnalyzer`,
    :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` and :py:class:`kinisi.analyze.ConductivityAnalyzer` classes.

    :param trajectory: The parsed trajectory from some input file. This will be of type :py:class:`Parser`, but
        the specifics depend on the parser that is used.
    """

    def __init__(self, trajectory: Parser) -> None:
        self.trajectory = trajectory

    @classmethod
    def _from_xdatcar(
        cls,
        trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar', list['pymatgen.io.vasp.outputs.Xdatcar']],
        specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
        time_step: sc.Variable,
        step_skip: sc.Variable,
        dtype: str | None = None,
        dt: sc.Variable = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: sc.Variable = None,
        masses: sc.Variable = None,
        progress: bool = True,
    ) -> 'Analyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from a single or a list of
        :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects.

        :param trajectory: The :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or list of these that should be parsed.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note,
            that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit
            that is use for the time interval values.
        :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
            output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
            a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
        :param dtype: If :py:attr:`trajectory` is a :py:class:`pymatgen.io.vasp.outputs.Xdatcar` object, this should
            be :py:attr:`None`. However, if a list of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects is passed,
            then it is necessary to identify if these constitute a series of :py:attr:`consecutive` trajectories or
            a series of :py:attr:`identical` starting points with different random seeds, in which case the `dtype`
            should be either :py:attr:`consecutive` or :py:attr:`identical`.:
        :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
            ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with
            a step size the same as the smallest interval.
        :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
            the axes of interest. Optional, defaults to `'xyz'`.
        :param distance_unit: The unit of distance in the simulation input. This should be a :py:mod:`scipp` unit and
            defaults to :py:attr:`sc.units.angstrom`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
        """
        if dtype is None:
            p = PymatgenParser(
                trajectory.structures,
                specie,
                time_step,
                step_skip,
                dt,
                dimension,
                distance_unit,
                specie_indices,
                masses,
                progress,
            )
            return cls(p)
        elif dtype == 'identical':
            u = [
                PymatgenParser(
                    f.structures,
                    specie,
                    time_step,
                    step_skip,
                    dt,
                    dimension,
                    distance_unit,
                    progress,
                    specie_indices,
                    masses,
                )
                for f in trajectory
            ]
            p = u[0]
            p.displacements = sc.concat([i.displacements for i in u], 'atom')
            return cls(p)
        elif dtype == 'consecutive':
            structures = _flatten_list([x.structures for x in trajectory])
            p = PymatgenParser(
                structures, specie, time_step, step_skip, dt, dimension, distance_unit, specie_indices, masses, progress
            )
            return cls(p)

    @classmethod
    def _from_universe(
        cls,
        trajectory: 'MDAnalysis.core.universe.Universe',
        specie: str,
        time_step: sc.Variable,
        step_skip: sc.Variable,
        dtype: str | None = None,
        dt: sc.Variable = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: sc.Variable = None,
        masses: sc.Variable = None,
        progress: bool = True,
    ) -> 'Analyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from a
        :py:class:`MDAnalysis.core.universe.Universe` object.
        """
        if dtype is None:
            p = MDAnalysisParser(
                universe=trajectory,
                specie=specie,
                time_step=time_step,
                step_skip=step_skip,
                dt=dt,
                dimension=dimension,
                distance_unit=distance_unit,
                specie_indices=specie_indices,
                masses=masses,
                progress=progress,
            )
            print(p)
            return cls(p)
        elif dtype == 'identical':
            u = [
                MDAnalysisParser(
                    universe=f,
                    specie=specie,
                    time_step=time_step,
                    step_skip=step_skip,
                    dt=dt,
                    dimension=dimension,
                    distance_unit=distance_unit,
                    specie_indices=specie_indices,
                    masses=masses,
                    progress=progress,
                )
                for f in trajectory
            ]
            p = u[0]
            p.displacements = sc.concat([i.displacements for i in u], 'atom')
            return cls(p)

    @classmethod
    def _from_ase(
        cls,
        trajectory: Union['ase.io.trajectory.Trajectory', list['ase.io.trajectory.Trajectory']],
        specie: Union[str, 'ase.Atom', None],
        time_step: sc.Variable,
        step_skip: sc.Variable,
        dtype: str | None = None,
        dt: sc.Variable = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: sc.Variable = None,
        masses: sc.Variable = None,
        progress: bool = True,
    ) -> 'Analyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from a single or a list of
        :py:class:`ase.io.trajectory.Trajectory` objects.
        """
        if dtype is None:
            p = ASEParser(
                trajectory, specie, time_step, step_skip, dt, dimension, distance_unit, specie_indices, masses, progress
            )
            return cls(p)
        elif dtype == 'identical':
            u = [
                ASEParser(
                    f, specie, time_step, step_skip, dt, dimension, distance_unit, specie_indices, masses, progress
                )
                for f in trajectory
            ]
            p = u[0]
            p.displacements = sc.concat([i.displacements for i in u], 'atom')
            return cls(p)

    def posterior_predictive(
        self, n_posterior_samples: int = None, n_predictive_samples: int = 256, progress: bool = True
    ):
        """
        Sample  the posterior predictive distribution. The shape of the resulting array will be
        `(n_posterior_samples * n_predictive_samples, start_dt)`.

        :params posterior_predictive_params: Parameters for the :py:func:`diffusion.posterior_predictive` method.
            See the appropriate documentation for more guidence on this dictionary.

        :return: Samples from the posterior predictive distribution.
        """
        return self.diff.posterior_predictive(n_posterior_samples, n_predictive_samples, progress)

    @property
    def n_atoms(self) -> int:
        """
        :returns: The number of atoms in the trajectory.
        """
        return self.trajectory.displacements.sizes['atom']

    @property
    def intercept(self) -> sc.Variable:
        """
        :return: The intercept of the linear relationship.
        """
        return self.diff.intercept

    @property
    def dt(self) -> sc.Variable:
        """
        :return: The time intervals used for the mean-squared displacement.
        """
        return self._da.coords['time interval']

    @property
    def da(self) -> sc.DataArray:
        """
        :return: The mean-squared displacement data array.
        """
        return self._da


def _flatten_list(this_list: list) -> list:
    """
    Flatten nested lists.

    :param this_list: List to be flattened

    :return: Flattened list
    """
    return [item for sublist in this_list for item in sublist]


def _stack_trajectories(u: sc.Variable) -> list[np.ndarray]:
    """
    If more than one trajectory is given, then they are stacked to give the appearance that there are
    additional atoms in the trajectory.

    :param u: Results from the parsing of each trajectory.

    :return: The stacked displacement list.
    """
    joint_disp_3d = []
    for i in range(len(u[0].disp_3d)):
        disp = np.zeros((u[0].disp_3d[i].shape[0] * len(u), u[0].disp_3d[i].shape[1], u[0].disp_3d[i].shape[2]))
        disp[: u[0].disp_3d[i].shape[0]] = u[0].disp_3d[i]
        for j in range(1, len(u)):
            disp[u[0].disp_3d[i].shape[0] * j : u[0].disp_3d[i].shape[0] * (j + 1)] = u[j].disp_3d[i]
        joint_disp_3d.append(disp)
    return joint_disp_3d
