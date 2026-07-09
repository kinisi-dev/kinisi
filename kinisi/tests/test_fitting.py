"""
Tests for fitting module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

# pylint: disable=R0201

import unittest
import pytest

import numpy as np
import scipp as sc
from numpy.testing import assert_almost_equal, assert_equal
from scipy.stats import norm, uniform
from scipy.stats._distn_infrastructure import rv_frozen

from kinisi import fitting
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


class TestFittingBase(unittest.TestCase):
    """
    Unit tests for FittingBase class
    """

    def test_init(self):
        """
        Test the initialisation of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert_equal(td.function, straight_line)
        assert td.parameter_names == ('m', 'c')
        assert td.parameter_units == (sc.Unit('m/s'), sc.Unit('m'))
        assert isinstance(td.data_group['m'], sc.Variable)
        assert isinstance(td.data_group['c'], sc.Variable)
        assert isinstance(td.priors[0], rv_frozen)
        assert isinstance(td.priors[1], rv_frozen)

    def test_init_priors(self):
        """
        Test the initialisation of FittingBase class with priors
        """
        priors = [uniform(0, 1), uniform(0, 1e20)]
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), priors=priors)
        assert_equal(td.function, straight_line)
        assert td.parameter_names == ('m', 'c')
        assert td.parameter_units == (sc.Unit('m/s'), sc.Unit('m'))
        assert td.priors[0] == priors[0]
        assert td.priors[1] == priors[1]

    def test_init_norm_priors(self):
        """
        Test the initialisation of the FittingBase Class
        """
        priors = [norm(10, 1), norm(0, 1)]
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), priors=priors)
        assert_equal(td.function, straight_line)
        assert td.parameter_names == ('m', 'c')
        assert td.parameter_units == (sc.Unit('m/s'), sc.Unit('m'))
        assert td.priors[0] == priors[0]
        assert td.priors[1] == priors[1]

    def test_init_wrong_priors(self):
        """
        Test the initialisation of FittingBase class with wrong number of priors
        """
        with self.assertRaises(ValueError):
            priors = [uniform(0, 1), uniform(0, 1e20), uniform(0, 1)]
            fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), priors=priors)

    def test_repr(self):
        """
        Test the string representation of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert str(td.__repr__()) == str(td.data_group.__repr__())

    def test_str(self):
        """
        Test the string representation of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert str(td) == str(td.data_group)

    def test_repr_html(self):
        """
        Test the HTML representation of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td._repr_html_(), str)

    def test_log_likelihood(self):
        """
        Test the log-likelihood function of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.log_likelihood([1, 0]), float)
        assert_almost_equal(td.log_likelihood([1, 0]), -13.275855715784758)

    def test_nll(self):
        """
        Test the negative log-likelihood function of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.nll([1, 0]), float)
        assert_almost_equal(td.nll([1, 0]), 13.275855715784758)

    def test_log_prior(self):
        """
        Test the log-prior function of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        assert isinstance(td.log_prior([1, 0]), float)
        assert_almost_equal(td.log_prior([1, 0]), -np.inf)

    def test_mcmc(self):
        """
        Test the MCMC sampling function of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert td.data_group['m'].shape == (32,)
        assert td.data_group['c'].shape == (32,)
    
    def test_mcmc_x0(self):
        """
        Test the MCMC sampling function with an x0 value.
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(x0=(1 * sc.Unit('m/s'), 0.5 * sc.Unit('m')), n_samples=10, n_burn=5, n_walkers=32)
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert td.data_group['m'].shape == (32,)
        assert td.data_group['c'].shape == (32,) 

    def test_mcmc_x0_wrong_unit(self):
        """
        Test the MCMC sampling function with an x0 value.
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        with pytest.raises(TypeError):
            td.mcmc(x0=(1 * sc.Unit('m'), 0.5 * sc.Unit('m')), n_samples=10, n_burn=5, n_walkers=32)

    def test_nested_sampling(self):
        """
        Test the nested sampling function of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.nested_sampling()
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert isinstance(td.logz, sc.Variable)

    def test_nested_sampling_norm_priors(self):
        """
        Test the nested sampling function of FittingBase class where priors are norm
        """
        td = fitting.FittingBase(
            data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')), priors=[norm(1, 1), norm(0, 1)]
        )
        td.nested_sampling()
        assert isinstance(td.data_group['m'], Samples)
        assert isinstance(td.data_group['c'], Samples)
        assert isinstance(td.logz, sc.Variable)

    def test_flatchain(self):
        """
        Test the flatchain property of FittingBase class
        """
        td = fitting.FittingBase(data, straight_line, ('m', 'c'), (sc.Unit('m/s'), sc.Unit('m')))
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        td.mcmc(n_samples=10, n_burn=5, n_walkers=32)
        assert isinstance(td.flatchain, sc.DataGroup)
        assert len(td.flatchain) == 2
        assert td.flatchain.shape == (32,)
