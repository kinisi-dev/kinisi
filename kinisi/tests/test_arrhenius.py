"""
Tests for arrhenius module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

# pylint: disable=R0201

import unittest

import scipp as sc
from numpy.testing import assert_almost_equal, assert_equal
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
    Unit tests for the TemperatureDependent class
    """

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

    def test_init_priros(self):
        """
        Test the initialisation of Arrhenius class with priors
        """
        priors = [uniform(0, 1), uniform(0, 1e20)]
        arr = arrhenius.Arrhenius(data, priors=priors)
        assert_equal(arr.function, arrhenius.arrhenius)
        assert arr.parameter_names == ('activation_energy', 'preexponential_factor')
        assert arr.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'))
        assert arr.priors[0] == priors[0]
        assert arr.priors[1] == priors[1]

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

    def test_init_priors(self):
        """
        Test the initialisation of VogelFulcherTammann class with priors
        """
        priors = (
            uniform(0, 1),
            uniform(0, 1e20),
            uniform(0, 1000),
        )
        vtf = arrhenius.VogelFulcherTammann(data, priors=priors)
        assert_equal(vtf.function, arrhenius.vtf_equation)
        assert vtf.parameter_names == ('activation_energy', 'preexponential_factor', 'T0')
        assert vtf.parameter_units == (sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'))
        assert vtf.priors[0] == priors[0]
        assert vtf.priors[1] == priors[1]
        assert vtf.priors[2] == priors[2]

    def test_vtf_equation(self):
        """
        Test the super arrhenius function
        """
        assert_almost_equal(9.995999241, arrhenius.vtf_equation(300, 1e-5, 10, 10), decimal=5)
