"""
The :py:class:`kinisi.analyze.DiffusionAnalyser` class enable the evaluation of tracer mean-squared
displacment and the self-diffusion coefficient.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

from typing import Union

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType

from kinisi.analyzer import Analyzer
from kinisi.diffusion import Diffusion
from kinisi.displacement import calculate_msd
from kinisi.parser import Parser


class DiffusionAnalyzer(Analyzer):
    """
    The class for the investigation of the self-diffusion.

    :param trajectory: The parsed trajectory from some input file. This will be of type :py:class:`Parser`, but
        the specifics depend on the parser that is used.
    :param da: The mean-squared displacement data, which is a :py:class:`scipp.DataArray` object.
        This is calculated from the trajectory data and is used to determine the diffusion coefficient.
    """

    def __init__(self, trajectory: Parser) -> None:
        super().__init__(trajectory)
        self._da = None

    @classmethod
    def from_xdatcar(
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
    ) -> 'DiffusionAnalyzer':
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
        :param specie_indices: The indices of the species to compute the centre of mass of. Optional, defaults to
            :py:attr:`None`, which means that all species are considered.
        :param masses: The masses of the species to calculate the diffusion for. Optional, defaults
            to :py:attr:`None`, which means that the masses are not considered.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :returns: The :py:class:`DiffusionAnalyzer` object with the mean-squared displacement calculated.
        """
        p = super()._from_xdatcar(
            trajectory,
            specie,
            time_step,
            step_skip,
            dtype,
            dt,
            dimension,
            distance_unit,
            specie_indices,
            masses,
            progress,
        )
        p._da = calculate_msd(p.trajectory, progress)
        return p

    @classmethod
    def from_universe(
        cls,
        trajectory: 'MDAnalysis.core.universe.Universe',
        specie: str = None,
        time_step: VariableLikeType = None,
        step_skip: VariableLikeType = None,
        dtype: str | None = None,
        dt: VariableLikeType = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: sc.Variable = None,
        masses: sc.Variable = None,
        progress: bool = True,
    ) -> 'DiffusionAnalyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from a
        :py:class:`MDAnalysis.Universe` object.

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
        :param specie_indices: The indices of the species to compute the centre of mass of. Optional, defaults to
            :py:attr:`None`, which means that all species are considered.
        :param masses: The masses of the species to calculate the diffusion for. Optional, defaults
            to :py:attr:`None`, which means that the masses are not considered.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :returns: The :py:class:`DiffusionAnalyzer` object with the mean-squared displacement calculated.
        """
        p = super()._from_universe(
            trajectory,
            specie,
            time_step,
            step_skip,
            dtype,
            dt,
            dimension,
            distance_unit,
            specie_indices,
            masses,
            progress,
        )
        p._da = calculate_msd(p.trajectory, progress)
        return p

    @classmethod
    def from_ase(
        cls,
        trajectory: Union['ase.io.trajectory.Trajectory', list['ase.io.trajectory.Trajectory']],
        specie: str = None,
        time_step: VariableLikeType = None,
        step_skip: VariableLikeType = None,
        dtype: str | None = None,
        dt: VariableLikeType = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: sc.Variable = None,
        masses: sc.Variable = None,
        progress: bool = True,
    ) -> 'DiffusionAnalyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from an ASE trajectory.

        :param trajectory: The ASE trajectory or list of ASE trajectories to parse.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator.
            Note, that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be
            the unit that is use for the time interval values.
        :param step_skip: Sampling frequency of the simulation trajectory, i.e., how many time steps exist between the
            output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
            a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
        :param dtype: If :py:attr:`trajectory` is a single ASE trajectory, this should be :py:attr:`None`. However, if
            a list of ASE trajectories is passed, then it is necessary to identify if these constitute a series of
            :py:attr:`consecutive` trajectories or a series of :py:attr:`identical` starting points with different
            random seeds, in which case the `dtype` should be either :py:attr:`consecutive` or :py:attr:`identical`.
        :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
            ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with
            a step size the same as the smallest interval.
        :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
            the axes of interest. Optional, defaults to `'xyz'`.
        :param distance_unit: The unit of distance in the simulation input. This should be a :py:mod:`scipp` unit and
            defaults to :py:attr:`sc.units.angstrom`.
        :param specie_indices: The indices of the species to compute the centre of mass of. Optional, defaults to
            :py:attr:`None`, which means that all species are considered.
        :param masses: The masses of the species to calculate the diffusion for. Optional, defaults
            to :py:attr:`None`, which means that the masses are not considered.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
        :returns: The :py:class:`DiffusionAnalyzer` object with the mean-squared displacement calculated.
        """
        p = super()._from_ase(
            trajectory=trajectory,
            specie=specie,
            time_step=time_step,
            step_skip=step_skip,
            dtype=dtype,
            dt=dt,
            dimension=dimension,
            distance_unit=distance_unit,
            specie_indices=specie_indices,
            masses= masses,
            progress=progress,
        )
        p._da = calculate_msd(p.trajectory, progress)
        return p

    def diffusion(
        self,
        start_dt: VariableLikeType,
        cond_max: float = 1e16,
        fit_intercept: bool = True,
        n_samples: int = 1000,
        n_walkers: int = 32,
        n_burn: int = 500,
        n_thin: int = 10,
        progress: bool = True,
        random_state: np.random.mtrand.RandomState = None,
    ) -> None:
        """
        Calculate the diffusion coefficient using the mean-squared displacement data.

        :param start_dt: The time at which the diffusion regime begins.
        :param cond_max: The maximum condition number of the covariance matrix. Optional, default is :py:attr:`1e16`.
        :param fit_intercept: Whether to fit an intercept. Optional, default is :py:attr:`True`.
        :param n_samples: The number of MCMC samples to take. Optional, default is :py:attr:`1000`.
        :param n_walkers: The number of walkers to use in the MCMC. Optional, default is :py:attr:`32`.
        :param n_burn: The number of burn-in samples to discard. Optional, default is :py:attr:`500`.
        :param n_thin: The thinning factor for the MCMC samples. Optional, default is :py:attr:`10`.
        :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.
        :param random_state: The random state to use for the MCMC. Optional, default is :py:attr:`None`.
        """
        self.diff = Diffusion(da=self._da)
        self.diff._diffusion(
            start_dt,
            cond_max=cond_max,
            fit_intercept=fit_intercept,
            n_samples=n_samples,
            n_walkers=n_walkers,
            n_burn=n_burn,
            n_thin=n_thin,
            progress=progress,
            random_state=random_state,
        )

    @property
    def distributions(self) -> np.array:
        """
        :return: A distribution of samples for the linear relationship that can be used for easy
        plotting of credible intervals.
        """
        if self.diff.intercept is not None:
            return (
                self.diff.gradient.values * self._da.coords['time interval'].values[:, np.newaxis]
                + self.diff.intercept.values
            )
        else:
            return self.diff.gradient.values * self._da.coords['time interval'].values[:, np.newaxis]

    @property
    def D(self) -> VariableLikeType:
        """
        :return: The self-diffusion coefficient.
        """
        return self.diff.D

    @property
    def msd(self) -> VariableLikeType:
        """
        :return: The mean-squared displacement.
        """
        return self._da.data

    @property
    def flatchain(self) -> sc.DataGroup:
        """
        :returns: The flatchain of the MCMC samples.
        """
        flatchain = {'D*': self.D}
        if self.intercept is not None:
            flatchain['intercept'] = self.intercept
        return sc.DataGroup(**flatchain)
