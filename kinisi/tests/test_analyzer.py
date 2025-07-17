"""
Tests for the analyzer module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest

import MDAnalysis as mda
import scipp as sc
from ase.io import Trajectory
from pymatgen.io.vasp import Xdatcar

import kinisi
from kinisi.analyzer import Analyzer, _flatten_list

file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
xd = Xdatcar(file_path)
da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('ps'), 'step_skip': 50 * sc.Unit('dimensionless')}
md = mda.Universe(
    os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
    os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
    format='LAMMPS',
)
db_params = {'specie': '1', 'time_step': 0.005 * sc.Unit('ps'), 'step_skip': 250 * sc.Unit('dimensionless')}
ase_file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase.traj')
traj = Trajectory(ase_file_path, 'r')
dc_params = {'specie': 'Li', 'time_step': 1.0 * 1e-3 * sc.Unit('fs'), 'step_skip': 1 * sc.Unit('dimensionless')}


class TestAnalyzer(unittest.TestCase):
    """
    Tests for the Analyzer class.
    """

    def test_to_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = Analyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        file_exists = os.path.exists(test_file)
        os.remove(test_file)
        assert file_exists

    def test_load_hdf5(self):
        test_file = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_Analyzer.h5')
        analyzer = Analyzer.from_hdf5(test_file)
        assert type(analyzer) is Analyzer

    def test_round_trip_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = Analyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        analyzer_2 = Analyzer.from_hdf5(test_file)
        if os.path.exists(test_file):
            os.remove(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)

    def test_xdatcar_pmg(self):
        a = Analyzer._from_xdatcar(xd, **da_params)
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (140, 192, 3)

    def test_identical_xdatcar_pmg(self):
        a = Analyzer._from_xdatcar([xd, xd], **da_params, dtype='identical')
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (140, 192 * 2, 3)

    def test_consecutive_xdatcar_pmg(self):
        a = Analyzer._from_xdatcar([xd, xd], **da_params, dtype='consecutive')
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (140 * 2, 192, 3)

    def test_mdauniverse(self):
        a = Analyzer._from_universe(md, **db_params)
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (200, 204, 3)

    def test_identical_mdauniverse(self):
        a = Analyzer._from_universe([md, md], **db_params, dtype='identical')
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (200, 204 * 2, 3)

    def test_ase(self):
        a = Analyzer._from_ase(traj, **dc_params)
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (200, 180, 3)

    def test_identical_ase(self):
        a = Analyzer._from_ase([traj, traj], **dc_params, dtype='identical')
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (200, 180 * 2, 3)

    def test_consecutive_ase(self):
        a = Analyzer._from_ase([traj, traj], **dc_params, dtype='consecutive')
        assert a.trajectory.displacements.dims == ('obs', 'atom', 'dimension')
        assert a.trajectory.displacements.shape == (200 * 2, 180, 3)

    def test_list_bad_input(self):
        with self.assertRaises(ValueError):
            _ = Analyzer._from_xdatcar([file_path, file_path], **da_params, dtype='consecutie')

    def test_list_bad_mda(self):
        with self.assertRaises(ValueError):
            _ = Analyzer._from_universe(file_path, **db_params, dtype='consecutie')


class TestFunctions(unittest.TestCase):
    """
    Tests for other functions
    """

    def test__flatten_list(self):
        a_list = [[1, 2, 3], [4, 5]]
        result = _flatten_list(a_list)
        assert result == [1, 2, 3, 4, 5]
