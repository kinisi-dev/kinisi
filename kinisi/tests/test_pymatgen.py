"""
Tests for the pymatgen module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest

import scipp as sc
from pymatgen.io.vasp import Xdatcar

import kinisi
from kinisi import parser
from kinisi.pymatgen import PymatgenParser


class TestPymatgenParser(unittest.TestCase):
    """
    Unit tests for the pymatgen module
    """

    def test_pymatgen_datagroup_round_trip(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        data = PymatgenParser(xd.structures, **da_params)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert vars(data) == vars(data_2)
        assert type(data) is type(data_2)
        data_3 = PymatgenParser._from_datagroup(datagroup)
        assert vars(data) == vars(data_3)
        assert type(data) is type(data_3)
