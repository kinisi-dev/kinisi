"""
Tests for arrhenius module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

# pylint: disable=R0201

import unittest

import numpy as np
import scipp as sc
from numpy.testing import assert_almost_equal, assert_equal
from scipp import testing
from scipy.stats import uniform

from kinisi import arrhenius
from kinisi.samples import Samples

temp = sc.linspace(start=5, stop=50, num=10, dim='temperature', unit=sc.units.K)
D = sc.linspace(start=5, stop=50, num=10, unit='cm^2 / s', dim='temperature')
D.variances = D.values * 0.1  # 10% uncertainty
data = sc.DataArray(data=D, coords={'temperature': temp})


def straight_line(x, m, c):
    """
    A simple linear function for testing purposes.

    :param x: The independent variable.
    :param m: The slope of the line.
    :param c: The y-intercept of the line.
    :return: The value of the linear function at x.
    """
    return m * x + c


class TestTemperatureDependent(unittest.TestCase):
    """
    Unit tests for TemperatureDependent class
    """

    def test_init(self):
        """
        Test the initialisation of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        u = uniform(loc=0, scale=1)
        assert_equal(td.function, straight_line)
        assert td.parameter_names == ('m', 'c')
        assert td.parameter_units == (sc.Unit('m/s'), sc.Unit('m'))
        assert isinstance(td.data_group['m'], sc.Variable)
        assert isinstance(td.data_group['c'], sc.Variable)
        assert isinstance(td.bounds[0][0], sc.Variable)
        assert isinstance(td.bounds[0][1], sc.Variable)
        assert isinstance(td.bounds[1][0], sc.Variable)
        assert isinstance(td.bounds[1][1], sc.Variable)
        assert isinstance(td.priors[0], type(u))
        assert isinstance(td.priors[1], type(u))

    def test_init_bounds(self):
        """
        Test the initialisation of TemperatureDependent class with bounds
        """
        bounds = ((0 * sc.Unit('m/s'), 1 * sc.Unit('m/s')), (0 * sc.Unit('m'), 1e20 * sc.Unit('m')))
        td = arrhenius.TemperatureDependent(
            data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), bounds=bounds
        )
        assert_equal(td.function, straight_line)
        assert td.parameter_names == ('m', 'c')
        assert td.parameter_units == (sc.Unit('m/s'), sc.Unit('m'))
        testing.assert_allclose(td.bounds[0][0], bounds[0][0])
        testing.assert_allclose(td.bounds[0][1], bounds[0][1])
        testing.assert_allclose(td.bounds[1][0], bounds[1][0])
        testing.assert_allclose(td.bounds[1][1], bounds[1][1])

    def test_init_wrong_bounds(self):
        """
        Test the initialisation of TemperatureDependent class with wrong bounds
        """
        with self.assertRaises(ValueError):
            arrhenius.TemperatureDependent(
                data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), bounds=((0, 1), (0, 1e20), (0, 1))
            )

    def test_repr(self):
        """
        Test the string representation of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert str(td.__repr__()) == str(td.data_group.__repr__())

    def test_str(self):
        """
        Test the string representation of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert str(td) == str(td.data_group)

    def test_repr_html(self):
        """
        Test the HTML representation of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td._repr_html_(), str)

    def test_log_likelihood(self):
        """
        Test the log-likelihood function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.log_likelihood([1, 0]), float)
        assert_almost_equal(td.log_likelihood([1, 0]), -13.275855715784758)

    def test_nll(self):
        """
        Test the negative log-likelihood function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.nll([1, 0]), float)
        assert_almost_equal(td.nll([1, 0]), 13.275855715784758)

    def test_log_prior(self):
        """
        Test the log-prior function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.log_prior([1, 0]), float)
        assert_almost_equal(td.log_prior([1, 0]), -np.inf)

    def test_mcmc(self):
        """
        Test the MCMC sampling function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert td.data_group['m'].shape == (32,)
        assert td.data_group['c'].shape == (32,)

    def test_nested_sampling(self):
        """
        Test the nested sampling function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.nested_sampling()
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert isinstance(td.logz, sc.Variable)

    def test_flatchain(self):
        """
        Test the flatchain property of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        assert isinstance(td.flatchain, sc.DataGroup)
        assert len(td.flatchain) == 2
        assert td.flatchain.shape == (32,)

    def test_extrapolate(self):
        """
        Test the extrapolate function of TemperatureDependent class
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        extrapolated_value = td.extrapolate(300 * sc.Unit('K'))
        assert isinstance(extrapolated_value, sc.Variable)
        assert extrapolated_value.unit == sc.Unit('cm^2 / s')

    def test_extrapolate_mcmc(self):
        """
        Test the extrapolate function of TemperatureDependent class with MCMC
        """
        td = arrhenius.TemperatureDependent(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        extrapolated_value = td.extrapolate(300 * sc.Unit('K'))
        assert isinstance(extrapolated_value, Samples)
        assert extrapolated_value.unit == sc.Unit('cm^2 / s')
        assert_almost_equal(extrapolated_value.values.shape, (32,))


class TestArrhenius(unittest.TestCase):
    """
    Unit tests for Arrhenius class
    """

    def test_init(self):
        """
        Test the initialisation of Arrhenius class
        """
        arr = arrhenius.Arrhenius(data)
        assert_equal(arr.function, arrhenius.arrhenius)
        assert arr.parameter_names == ('activation_energy', 'preexponential_factor')
        assert arr.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'))
        assert isinstance(arr.activation_energy, sc.Variable)
        assert isinstance(arr.preexponential_factor, sc.Variable)

    def test_init_bounds(self):
        """
        Test the initialisation of Arrhenius class with bounds
        """
        bounds = ((0 * sc.Unit('eV'), 1 * sc.Unit('eV')), (0 * sc.Unit('cm^2/s'), 1e20 * sc.Unit('cm^2/s')))
        arr = arrhenius.Arrhenius(data, bounds=bounds)
        assert_equal(arr.function, arrhenius.arrhenius)
        assert arr.parameter_names == ('activation_energy', 'preexponential_factor')
        assert arr.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'))
        testing.assert_allclose(arr.bounds[0][0], bounds[0][0])
        testing.assert_allclose(arr.bounds[0][1], bounds[0][1])
        testing.assert_allclose(arr.bounds[1][0], bounds[1][0])
        testing.assert_allclose(arr.bounds[1][1], bounds[1][1])

    def test_arrhenius(self):
        """
        Test the arrhenius function
        """
        assert_almost_equal(9.996132574, arrhenius.arrhenius(300, 1e-5, 10), decimal=5)


class TestVTF(unittest.TestCase):
    """
    Unit tests for VogelFulcherTammann (VTF) class
    """

    def test_init(self):
        """
        Test the initialisation of VTF class
        """
        vtf = arrhenius.VogelFulcherTammann(data)
        assert_equal(vtf.function, arrhenius.vtf_equation)
        assert vtf.parameter_names == ('activation_energy', 'preexponential_factor', 'T0')
        assert vtf.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'))
        assert isinstance(vtf.activation_energy, sc.Variable)
        assert isinstance(vtf.preexponential_factor, sc.Variable)
        assert isinstance(vtf.T0, sc.Variable)

    def test_init_bounds(self):
        """
        Test the initialisation of VogelFulcherTammann class with bounds
        """
        bounds = (
            (0 * sc.Unit('eV'), 1 * sc.Unit('eV')),
            (0 * sc.Unit('cm^2/s'), 1e20 * sc.Unit('cm^2/s')),
            (0 * sc.Unit('K'), 1000 * sc.Unit('K')),
        )
        vtf = arrhenius.VogelFulcherTammann(data, bounds=bounds)
        assert_equal(vtf.function, arrhenius.vtf_equation)
        assert vtf.parameter_names == ('activation_energy', 'preexponential_factor', 'T0')
        assert vtf.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'))
        testing.assert_allclose(vtf.bounds[0][0], bounds[0][0])
        testing.assert_allclose(vtf.bounds[0][1], bounds[0][1])
        testing.assert_allclose(vtf.bounds[1][0], bounds[1][0])
        testing.assert_allclose(vtf.bounds[1][1], bounds[1][1])
        testing.assert_allclose(vtf.bounds[2][0], bounds[2][0])
        testing.assert_allclose(vtf.bounds[2][1], bounds[2][1])

    def test_vtf_equation(self):
        """
        Test the super arrhenius function
        """
        assert_almost_equal(9.995999241, arrhenius.vtf_equation(300, 1e-5, 10, 10), decimal=5)
