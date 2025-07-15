"""
Tests for the mdanalysis module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import unittest
import os

import scipp as sc
import MDAnalysis as mda

import kinisi
from kinisi import parser
from kinisi.mdanalysis import MDAnalysisParser

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipp.testing.assertions import assert_allclose, assert_identical

class TestMDAnalysisParser(unittest.TestCase):
    """
    Unit tests for the mdanalysis module
    """

    def test_mdanalysis_datagroup_round_trip(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                            os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                            format='LAMMPS')
        da_params = {'specie': '1', 'time_step': 0.005 * sc.Unit('fs'), 'step_skip': 250 * sc.Unit('dimensionless')}
        data = MDAnalysisParser(xd, **da_params)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert vars(data) == vars(data_2)
        assert type(data) is type(data_2)
        data_3 = MDAnalysisParser._from_datagroup(datagroup)
        assert vars(data) == vars(data_3)
        assert type(data) is type(data_3)

    def test_mda_init(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        da_params = {'specie': '1', 'time_step': 0.005, 'step_skip': 250}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_identical(data.indices, sc.array(dims=['atom'],values=list(range(204)),unit=sc.units.dimensionless))

    def test_mda_init_with_indices(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        specie_indices = sc.array(dims=['atom'],values=[208, 212],unit=sc.units.dimensionless)
        da_params = {'specie': None, 'time_step': 0.005, 'step_skip': 250, 'specie_indices': specie_indices}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_allclose(data.indices, sc.array(dims=['atom'],values=[208, 212],unit=sc.units.dimensionless))

    def test_mda_init_with_molecules(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        molecules = sc.array(dims=['group_of_atoms','atom'],values=[[0, 1, 2], [3, 4, 5]],unit=sc.units.dimensionless)
        da_params = {'specie': None, 'time_step': 0.005, 'step_skip': 250, 'specie_indices': molecules}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_allclose(data.coords['time',0]['atom',0], 
                        sc.array(dims=['time','atom','dimension'],
                                 values=np.array([[[0.6440246, 0.9255154, 0.3730706]]],dtype='float32'),
                                 unit=sc.units.dimensionless)['time',0]['atom',0])
        assert_allclose(data.indices, sc.array(dims=['atom'],values=list(range(len(molecules))),unit=sc.units.dimensionless))

    def test_mda_init_with_COG(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.traj'),
                          topology_format='DATA',
                          format='LAMMPSDUMP')
        specie_indices = sc.array(dims=['group_of_atoms','atom'],values=[[0, 1, 2]],unit=sc.units.dimensionless)
        da_params = {'specie': None, 'time_step': 1, 'step_skip': 1, 'specie_indices': specie_indices}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices.values, 0)
        assert_allclose(data.coords['time',0]['atom',0], 
                            sc.array(dims=['time','atom','dimension'],
                                     values=np.array([[[0.2, 0.2, 0.2]]],dtype='float32'),
                                     unit=sc.units.dimensionless)['time',0]['atom',0])

    # def test_mda_init_with_COM(self):
    #     xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.data'),
    #                       os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.traj'),
    #                       topology_format='DATA',
    #                       format='LAMMPSDUMP')
    #     da_params = {
    #         'specie': None,
    #         'time_step': 1,
    #         'step_skip': 1,
    #         'specie_indices': sc.array(dims=['group_of_atoms','atom'],values=[[1, 2, 3]],unit=sc.units.dimensionless),
    #         'masses': sc.array(dims=['group_of_atoms','mass'],values=[[1, 16, 1]],unit=sc.units.kg)
    #     }
    #     data = MDAnalysisParser(xd, **da_params)
    #     assert_almost_equal(data.time_step, 1)
    #     assert_almost_equal(data.step_skip, 1)
    #     assert_equal(data.indices, [0])
    #     assert_allclose(data.coords['time',0]['atom',0], 
    #                         sc.array(dims=['time','atom','dimension'],
    #                                  values=np.array([[[0.3788889, 0.2111111, 0.2]]],dtype='float32'),
    #                                  unit=sc.units.dimensionless)['time',0]['atom',0])

    # def test_mda_init_with_drift_indices(self):
    #     xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_drift.data'),
    #                       os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_drift.traj'),
    #                       topology_format='DATA',
    #                       format='LAMMPSDUMP')
    #     da_1_params = {'specie': '1', 'time_step': 1, 'step_skip': 1, 'drift_indices': []}
    #     data_1 = MDAnalysisParser(xd, **da_1_params)
    #     assert_almost_equal(data_1.time_step, 1)
    #     assert_almost_equal(data_1.step_skip, 1)
    #     assert_equal(data_1.indices, [0])
    #     assert_equal(data_1.drift_indices, [])
    #     assert_equal(data_1.dc[0], np.zeros((4, 3)))
    #     da_2_params = {'specie': '1', 'time_step': 1, 'step_skip': 1}
    #     data_2 = MDAnalysisParser(xd, **da_2_params)
    #     assert_almost_equal(data_2.time_step, 1)
    #     assert_almost_equal(data_2.step_skip, 1)
    #     assert_equal(data_2.indices, [0])
    #     assert_equal(data_2.drift_indices, list(range(1, 9)))
    #     print(data_2.dc[0])
    #     disp_array = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [1, 1, 1]]
    #     assert_almost_equal(data_2.dc[0], disp_array, decimal=6)