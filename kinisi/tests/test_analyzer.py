"""
Tests for the analyzer module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import unittest
import os

import scipp as sc
from pymatgen.io.vasp import Xdatcar

import kinisi
from kinisi.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    """
    Tests for the Analyzer class.
    """

    def test_to_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = Analyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer._to_hdf5(test_file)
        file_exists = os.path.exists(test_file)
        os.remove(test_file)
        assert file_exists
    
    def test_load_hdf5(self):
        test_file = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_Analyzer.h5')
        analyzer = Analyzer._from_hdf5(test_file)
        assert type(analyzer) is Analyzer
    
    def test_round_trip_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = Analyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer._to_hdf5(test_file)
        analyzer_2 = Analyzer._from_hdf5(test_file)
        if os.path.exists(test_file):
            os.remove(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)