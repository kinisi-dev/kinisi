"""
Tests for the pymatgen module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import unittest

import numpy as np
import scipp as sc
from pymatgen.io.vasp import Xdatcar

from kinisi import parser
from kinisi.pymatgen import PymatgenParser
from kinisi.tests import TEST_FILE_PATH


class TestPymatgenParser(unittest.TestCase):
    """
    Unit tests for the pymatgen module
    """

    def test_pymatgen_datagroup_round_trip(self):
        xd = Xdatcar(TEST_FILE_PATH / 'example_XDATCAR.gz')
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        data = PymatgenParser(xd.structures, **da_params)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert vars(data) == vars(data_2)
        assert type(data) is type(data_2)
        data_3 = PymatgenParser._from_datagroup(datagroup)
        assert vars(data) == vars(data_3)
        assert type(data) is type(data_3)

    def test_get_species_indices_init(self):
        molecules = np.arange(0, 1500).reshape(-1, 12)
        specie_indices = sc.array(
            dims=['particle', 'atoms in particle'], values=molecules, unit=sc.Unit('dimensionless')
        )
        structure = np.zeros(1500)
        PymatgenParser.get_drift_indices('', structure, specie_indices)
