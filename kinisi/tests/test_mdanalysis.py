"""
Tests for the mdanalysis module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import unittest

import MDAnalysis as mda
import numpy as np
import scipp as sc
from numpy.testing import assert_almost_equal
from scipp.testing.assertions import assert_allclose, assert_identical

from kinisi import parser
from kinisi.mdanalysis import MDAnalysisParser
from kinisi.tests import TEST_FILE_PATH


class TestMDAnalysisParser(unittest.TestCase):
    """
    Unit tests for the mdanalysis module
    """

    def test_mdanalysis_datagroup_round_trip(self):
        xd = mda.Universe(
            TEST_FILE_PATH / 'example_LAMMPS.data',
            TEST_FILE_PATH / 'example_LAMMPS.dcd',
            format='LAMMPS',
        )
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
        xd = mda.Universe(
            TEST_FILE_PATH / 'example_LAMMPS.data',
            TEST_FILE_PATH / 'example_LAMMPS.dcd',
            format='LAMMPS',
        )
        da_params = {'specie': '1', 'time_step': 0.005, 'step_skip': 250}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_identical(
            data.indices, sc.array(dims=['particle'], values=list(range(204)), unit=sc.units.dimensionless)
        )

    def test_mda_init_with_indices(self):
        xd = mda.Universe(
            TEST_FILE_PATH / 'example_LAMMPS.data',
            TEST_FILE_PATH / 'example_LAMMPS.dcd',
            format='LAMMPS',
        )
        specie_indices = sc.array(dims=['particle'], values=[208, 212], unit=sc.units.dimensionless)
        da_params = {'specie': None, 'time_step': 0.005, 'step_skip': 250, 'specie_indices': specie_indices}
        data = MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_allclose(data.indices, sc.array(dims=['particle'], values=[208, 212], unit=sc.units.dimensionless))

    def test_get_species_indices_init(self):
        molecules = np.arange(0, 1500).reshape(-1, 12)
        specie_indices = sc.array(
            dims=['particle', 'atoms in particle'], values=molecules, unit=sc.Unit('dimensionless')
        )
        structure = np.zeros(1500)
        MDAnalysisParser.get_drift_indices('', structure, specie_indices)
