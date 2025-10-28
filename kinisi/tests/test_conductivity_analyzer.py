"""
Tests for the conductivity_analyzer module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest
import warnings

import MDAnalysis as mda
import scipp as sc
from ase.io import Trajectory
from numpy.testing import assert_almost_equal
from pymatgen.io.vasp import Xdatcar
from scipp.testing import assert_allclose

from kinisi.analyze import ConductivityAnalyzer
from kinisi.analyzer import Analyzer
from kinisi.samples import Samples
from kinisi.tests import TEST_FILE_PATH

file_path = TEST_FILE_PATH / 'example_XDATCAR.gz'
xd = Xdatcar(file_path)
da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('ps'), 'step_skip': 50 * sc.Unit('dimensionless')}

ase_file_path = TEST_FILE_PATH / 'example_ase.traj'
traj = Trajectory(ase_file_path, 'r')
ase_params = {'specie': 'Li', 'time_step': 1.0 * 1e-3 * sc.Unit('fs'), 'step_skip': 1 * sc.Unit('dimensionless')}

mda_universe = mda.Universe(
    TEST_FILE_PATH / 'example_LAMMPS.data',
    TEST_FILE_PATH / 'example_LAMMPS.dcd',
    format='LAMMPS',
)
mda_params = {'specie': '1', 'time_step': 0.005 * sc.Unit('fs'), 'step_skip': 250 * sc.Unit('dimensionless')}


class TestConductivityAnalyzer(unittest.TestCase):
    """
    Tests for the ConductivityAnalyzer class.
    """

    def test_to_hdf5(cls):
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = ConductivityAnalyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        file_exists = os.path.exists(test_file)
        os.remove(test_file)
        assert file_exists

    def test_load_hdf(cls):
        test_file = TEST_FILE_PATH / 'example_DiffusionAnalyzer.h5'
        analyzer = ConductivityAnalyzer.from_hdf5(test_file)
        analyzer_2 = Analyzer.from_hdf5(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)

    def test_round_trip_hdf5(self):
        xd = Xdatcar(TEST_FILE_PATH / 'example_XDATCAR.gz')
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = ConductivityAnalyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        analyzer_2 = ConductivityAnalyzer.from_hdf5(test_file)
        analyzer_3 = Analyzer.from_hdf5(test_file)
        if os.path.exists(test_file):
            os.remove(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)
        assert analyzer.trajectory._to_datagroup() == analyzer_3.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_3)

    def test_properties(self):
        with warnings.catch_warnings(record=True) as _:
            a = ConductivityAnalyzer.from_xdatcar(xd, ionic_charge=1 * sc.Unit('e'), **da_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as _:
            a = ConductivityAnalyzer.from_xdatcar(xd, ionic_charge=1 * sc.Unit('e'), **da_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)
            a.conductivity(0 * sc.Unit('ps'), temperature=100 * sc.Unit('K'))
            assert isinstance(a.sigma, Samples)
            assert a.sigma.unit == sc.Unit('mS/cm')
            assert a.flatchain['sigma'].shape == (3200,)
            assert a.flatchain['intercept'].shape == (3200,)

    def test_from_ase(self):
        """Test creating ConductivityAnalyzer from ASE trajectory."""
        with warnings.catch_warnings(record=True) as _:
            a = ConductivityAnalyzer.from_ase(trajectory=traj, ionic_charge=1 * sc.Unit('e'), **ase_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)
            a.conductivity(0 * sc.Unit('fs'), temperature=100 * sc.Unit('K'))
            assert isinstance(a.sigma, Samples)
            assert a.sigma.unit == sc.Unit('mS/cm')
            assert a.flatchain['sigma'].shape == (3200,)
            assert a.flatchain['intercept'].shape == (3200,)

    def test_from_universe(self):
        """Test creating ConductivityAnalyzer from MDAnalysis Universe."""
        with warnings.catch_warnings(record=True) as _:
            a = ConductivityAnalyzer.from_universe(trajectory=mda_universe, ionic_charge=1 * sc.Unit('e'), **mda_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)
            a.conductivity(0 * sc.Unit('fs'), temperature=100 * sc.Unit('K'))
            assert isinstance(a.sigma, Samples)
            assert a.sigma.unit == sc.Unit('mS/cm')
            assert a.flatchain['sigma'].shape == (3200,)
            assert a.flatchain['intercept'].shape == (3200,)
