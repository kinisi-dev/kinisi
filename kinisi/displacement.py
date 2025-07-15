"""
Functions for the calculation of different types of displacement.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

from warnings import warn

import numpy as np
import scipp as sc
from tqdm import tqdm

from kinisi import parser


def calculate_msd(p: parser.Parser, progress: bool = True) -> sc.Variable:
    """
    Calculate the mean-squared displacement.

    :param p: The parser object containing the the relevant simulation trajectory data.
    :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.

    :return: A :py:class:`scipp.DataArray` object containing the relevant mean-squared displacement
        data and number of independent samples.
    """
    msd = []
    msd_var = []
    n_samples = []
    iterator = p.dt_index.values
    if progress:
        iterator = tqdm(p.dt_index.values, desc='Finding Means and Variances')
    for di in iterator:
        disp = sc.concat(
            [p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]], 'obs'
        )
        n = (p.displacements.sizes['atom'] * p.dt_index['time interval', -1] / di).value
        s = sc.sum(disp**2, 'dimension')
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        msd.append(m)
        msd_var.append(v)
        n_samples.append(n)

    return sc.DataArray(
        data=sc.Variable(dims=['time interval'], values=np.array(msd, dtype='float64'), variances=msd_var, unit=s.unit),
        coords={
            'time interval': p.dt,
            'n_samples': sc.array(dims=['time interval'], values=n_samples),
            'dimensionality': p.dimensionality,
        },
    )


def calculate_mstd(
    p: parser.Parser, system_particles: int = 1, ionic_charge: sc.Variable = None, progress: bool = True
) -> sc.Variable:
    """
    Calculate the mean-squared total displacement, i.e., the displacement of the centre-of-mass of all particles.

    :param p: The parser object containing the the relevant simulation trajectory data.
    :param system_particles: The number of system particles to average over. Note that the constitution of the
        system particles are defined in index order, i.e., two system particles will involve splitting the
        particles down the middle into each. Optional, defaults to :py:attr:`1`.
    :param ionic_charge: The ionic charge of the species of interest. This should be either a :py:mod:`scipp`
        scalar if all of the ions have the same charge or an array of the charge for each indiviudal ion.
        Optional, defaults to :py:attr:`None`, which means that the mean-squared total displacement is
        computed.
    :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.

    :return: A :py:class:`scipp.DataArray` object containing the relevant mean-squared total displacement
        data and number of independent samples.
    """
    mstd = []
    mstd_var = []
    n_samples = []
    iterator = p.dt_index.values
    if progress:
        iterator = tqdm(p.dt_index.values, desc='Finding Means and Variances')
    for di in iterator:
        disp = sc.concat(
            [p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]], 'obs'
        )
        disp = _consolidate_system_particles(disp, system_particles)
        n = (disp.sizes['atom'] * p.dt_index['time interval', -1] / di).value
        if ionic_charge is not None:
            disp = disp * ionic_charge
        s = sc.sum(disp**2, 'dimension')
        if s.size <= 1:
            continue
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        mstd.append(np.float64(m))
        mstd_var.append(np.float64(v))
        n_samples.append(n)

    return sc.DataArray(
        data=sc.Variable(dims=['time interval'], values=mstd, variances=mstd_var, unit=s.unit),
        coords={
            'time interval': p.dt['time interval', : len(mstd)],
            'n_samples': sc.array(dims=['time interval'], values=n_samples),
            'dimensionality': p.dimensionality,
        },
    )


def _consolidate_system_particles(disp: sc.DataArray, system_particles: int = 1) -> sc.DataArray:
    """
    Consolidate the displacement data to the specified number of centres of mass.

    :param disp: The displacement data to consolidate.
    :param system_particles: The number of centres of mass to average over. Note that the centres of mass are defined
        in index order, i.e., two centres of mass will split the atoms down the middle. Optional, defaults to :py:attr:`1`.

    :return: A :py:class:`scipp.DataArray` object containing the consolidated displacement data.
    """
    atoms_per_com = disp.sizes['atom'] // system_particles
    max_atoms = atoms_per_com * system_particles

    if max_atoms < disp.sizes['atom']:
        warn(
            f'Truncating {disp.sizes["atom"]} atoms to split evenly into {system_particles} centres of mass. '
            + 'This approach is inefficient, you should consider using the number of system particles to split this evenly.',
            stacklevel=2,
        )

    trimmed = disp['atom', :max_atoms]
    reshaped = trimmed.fold(dim='atom', sizes={'atom': system_particles, 'local': atoms_per_com})
    centres_of_mass = sc.sum(reshaped, dim='local')

    return centres_of_mass
