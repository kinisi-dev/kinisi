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