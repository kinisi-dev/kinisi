# """
# Tests for diffusion module

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

# Distributed under the terms of the MIT License

# @author: Andrew R. McCluskey (arm61) & Harry Richardson (Harry-Rich)
# """

import os
import unittest

import numpy as np
import pytest
import scipp as sc

import kinisi
from kinisi.diffusion import Diffusion, _straight_line, minimum_eigenvalue_method

# Random seed setting not yet implemented into bayesian regression and so cannot almost_equal


class TestFunctions(unittest.TestCase):
    """
    Testing other functions.
    """

    def test_minimum_eigenvalue_method(self):
        matrix = np.random.random((100, 100))
        reconditioned_matrix = minimum_eigenvalue_method(matrix, 1)
        assert not np.allclose(matrix, reconditioned_matrix)

    def test_straight_line(self):
        m = 3
        x = np.array([1, 2, 3])
        c = 1.3
        result = _straight_line(x, m, c)
        expected_result = np.array([4.3, 7.3, 10.3])
        assert np.all(result == expected_result)


class TestDiffusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.msd_load = sc.io.load_hdf5(
            filename=os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_msd.hdf5')
        )
        cls.RNG = np.random.RandomState(42)
        cls.diff = Diffusion(da=cls.msd_load)

    def test_bayesian_regression(self):
        start_dt = 10 * sc.Unit('femtosecond')
        self.diff.bayesian_regression(start_dt=start_dt, random_state=self.RNG)

        assert self.diff.gradient is not None
        assert hasattr(self.diff.gradient, 'mean')
        assert self.diff.gradient.to_unit('cm2/s').unit == sc.Unit('cm2/s')

        assert self.diff.intercept is not None
        assert self.diff.intercept.to_unit('angstrom2').unit == sc.Unit('angstrom2')

        assert self.diff._flatchain.shape == (3200, 2)

        cov = self.diff.covariance_matrix
        assert cov.dims == ('time_interval1', 'time_interval2')
        assert cov.shape == (140, 140)

    def test__diffusion(self):
        start_dt = 400 * sc.Unit('femtosecond')
        self.diff._diffusion(start_dt)

        assert self.diff._diffusion_coefficient.to_unit('cm2/s').unit == sc.Unit('cm2/s')
        assert self.diff.D.to_unit('cm2/s').unit == sc.Unit('cm2/s')

    def test__jump_diffusion(self):
        start_dt = 200 * sc.Unit('femtosecond')
        self.diff._jump_diffusion(start_dt)

        assert self.diff._jump_diffusion_coefficient.to_unit('cm2/s').unit == sc.Unit('cm2/s')
        assert self.diff.D_J.to_unit('cm2/s').unit == sc.Unit('cm2/s')

    def test__conductivity(self):
        diff_cond = Diffusion(da=self.msd_load)
        diff_cond.da = diff_cond.da * sc.scalar(1.0, unit=sc.Unit('coulomb2'))
        start_dt = 300 * sc.Unit('femtosecond')
        temp = 320 * sc.Unit('kelvin')
        volume = 300 * sc.Unit('angstrom3')

        diff_cond._conductivity(start_dt=start_dt, temperature=temp, volume=volume)

        assert diff_cond._sigma.to_unit('S/m').unit == 'S/m'
        assert diff_cond.sigma.to_unit('S/m').unit == 'S/m'

    def test_compute_covariance_matrix(self):
        start_dt = 200 * sc.Unit('femtosecond')
        self.diff.bayesian_regression(start_dt=start_dt)

        cov = self.diff.compute_covariance_matrix()

        # Basic structure checks
        assert isinstance(cov, sc.Variable)
        assert cov.dims == ('time_interval1', 'time_interval2')
        assert cov.shape[0] == cov.shape[1]
        assert cov.unit == self.msd_load.unit**2

        cov_array = cov.values
        assert np.allclose(cov_array, cov_array.T, atol=1e-10), 'Covariance matrix is not symmetric'
        assert np.any(np.diag(cov_array) != 0), 'Diagonal of covariance matrix contains only zeros'

    def test_posterior_predictive(self):
        post = self.diff.posterior_predictive()

        assert isinstance(post, sc.Variable)
        assert post.dims == ('samples', 'time interval')
        assert post.sizes['samples'] > 0
        assert post.sizes['time interval'] > 0
        assert np.all(np.isfinite(post.values))
        assert sc.to_unit(post, 'angstrom2').unit == sc.Unit('angstrom2')

        custom_samp = self.diff.posterior_predictive(n_posterior_samples=1, n_predictive_samples=400)
        assert custom_samp.dims == ('samples', 'time interval')
        assert custom_samp.sizes['samples'] == 400

        custom_samp2 = self.diff.posterior_predictive(n_posterior_samples=1, n_predictive_samples=400, progress=False)
        assert custom_samp2.dims == ('samples', 'time interval')

        diff_exc = Diffusion(da=self.msd_load)
        diff_exc.da = diff_exc.da * sc.scalar(1.0, unit=sc.Unit('watts'))

    def test_posterior_predictive_error(self):
        diff_exc = Diffusion(da=self.msd_load)
        start_dt = 400 * sc.Unit('femtosecond')
        diff_exc._diffusion(start_dt)
        diff_exc._covariance_matrix = diff_exc._covariance_matrix * sc.scalar(1.0, unit=sc.Unit('angstrom6'))

        with pytest.raises(ValueError, match='Units of the covariance matrix and mu do not align correctly'):
            diff_exc.posterior_predictive(n_posterior_samples=1, n_predictive_samples=400)
