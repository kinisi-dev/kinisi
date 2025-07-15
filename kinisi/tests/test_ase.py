"""
Tests for the ase module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest

import scipp as sc
from ase.io import Trajectory

import kinisi
from kinisi import parser
from kinisi.ase import ASEParser

class TestASEParser(unittest.TestCase):
    """
    Unit tests for the pymatgen module
    """

    def test_ase_datagroup_round_trip(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase.traj'))
        da_params = {'specie': 'Li', 'time_step': 1e-3 * sc.Unit('fs'), 'step_skip': 1 * sc.Unit('dimensionless')}
        data = ASEParser(traj, **da_params)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert vars(data) == vars(data_2)
        assert type(data) is type(data_2)
        data_3 = ASEParser._from_datagroup(datagroup)
        assert vars(data) == vars(data_3)
        assert type(data) is type(data_3)