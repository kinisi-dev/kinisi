"""
Tests for the conductivity_analyzer module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Oskar G. Soulas (osoulas)

import os
import unittest
import warnings

from numpy.testing import assert_almost_equal
import scipp as sc 
from scipp.testing import assert_allclose
from pymatgen.io.vasp import Xdatcar

import kinisi
from kinisi.analyzer import Analyzer
from kinisi.conductivity_analyzer import ConductivityAnalyzer
from kinisi.samples import Samples

file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
xd = Xdatcar(file_path)
da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('ps'), 'step_skip': 50 * sc.Unit('dimensionless')}


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
        test_file = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_DiffusionAnalyzer.h5')
        analyzer = ConductivityAnalyzer.from_hdf5(test_file)
        analyzer_2 = Analyzer.from_hdf5(test_file)
        assert analyzer.trajectory._to_datagroup() == analyzer_2.trajectory._to_datagroup()
        assert type(analyzer) is type(analyzer_2)

    def test_round_trip_hdf5(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
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
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer.from_xdatcar(xd,ionic_charge=1 * sc.Unit('e'), **da_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer.from_xdatcar(xd,ionic_charge=1 * sc.Unit('e'), **da_params)
            assert_allclose(a.dt, a.da.coords['time interval'])
            assert_almost_equal(a.mscd.values, a.da.values)
            assert_almost_equal(a.mscd.variances, a.da.variances)
            a.conductivity(0 * sc.Unit('ps'), temperature=100 * sc.Unit('K'))
            assert isinstance(a.sigma, Samples)
            print(a.sigma.unit)
            assert a.sigma.unit == sc.Unit('mS/cm')
            assert a.flatchain['sigma'].shape == (3200,)
            assert a.flatchain['intercept'].shape == (3200,)