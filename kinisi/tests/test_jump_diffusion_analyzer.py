"""
Tests for the jump_diffusion_analyzer module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest

import scipp as sc
from pymatgen.io.vasp import Xdatcar

import kinisi
from kinisi.analyzer import Analyzer
from kinisi.jump_diffusion_analyzer import JumpDiffusionAnalyzer


class TestJumpDiffusionAnalyzer(unittest.TestCase):
    """
    Tests for the JumpDiffusionAnalyzer class.
    """

    def test_to_hdf5(cls):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = JumpDiffusionAnalyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        file_exists = os.path.exists(test_file)
        os.remove(test_file)
        assert file_exists

    def test_load_hdf(cls):
        test_file = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_DiffusionAnalyzer.h5')
        analyzer = JumpDiffusionAnalyzer.from_hdf5(test_file)
        analyzer_2 = Analyzer.from_hdf5(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)

    def test_round_trip_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        analyzer = JumpDiffusionAnalyzer._from_xdatcar(xd, **da_params)
        test_file = 'test_save.h5'
        analyzer.to_hdf5(test_file)
        analyzer_2 = JumpDiffusionAnalyzer.from_hdf5(test_file)
        analyzer_3 = Analyzer.from_hdf5(test_file)
        if os.path.exists(test_file):
            os.remove(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)
        assert analyzer.trajectory._to_datagroup() == analyzer_3.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_3)
