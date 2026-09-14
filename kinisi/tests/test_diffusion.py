"""
Tests for diffusion module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# @author: Andrew R. McCluskey (arm61) & Harry Richardson (Harry-Rich)

import unittest
import warnings

import numpy as np
import pytest
import scipp as sc

from kinisi.diffusion import Diffusion, _straight_line, eigenvalue_clipping, marchenkopastur, minimum_eigenvalue_method
from kinisi.samples import Samples
from kinisi.tests import TEST_FILE_PATH

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

    def test_eigenvalue_clipping(self):
        matrix = np.random.random((100, 100)) + 100
        reconditioned_matrix = eigenvalue_clipping(matrix)
        assert not np.allclose(matrix, reconditioned_matrix)

    def test_marchenkopastur(self):
        x = np.linspace(1, 11, 10)
        result = marchenkopastur(x, 2, 2)
        actual = np.array([0.03978874, 0.02530364, 0.01740905, 0.01145199, 0.00519943, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, actual)


class TestDiffusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.msd_load = sc.io.load_hdf5(filename=TEST_FILE_PATH / 'example_msd2.hdf5')['_dg']
        cls.RNG = np.random.RandomState(42)
        cls.diff = Diffusion(dg=cls.msd_load)

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
        msd_load = sc.io.load_hdf5(filename=TEST_FILE_PATH / 'example_msd2.hdf5')['_dg']
        diff_cond = Diffusion(dg=msd_load)
        diff_cond.dg['da'] = diff_cond.dg['da'] * sc.scalar(1.0, unit=sc.Unit('coulomb2'))
        start_dt = 300 * sc.Unit('femtosecond')
        temp = 320 * sc.Unit('kelvin')
        volume = 300 * sc.Unit('angstrom3')

        diff_cond._conductivity(start_dt=start_dt, temperature=temp, volume=volume, number_of_particles=50)

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
        assert cov.unit == self.msd_load['da'].unit ** 2

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

        msd_load = sc.io.load_hdf5(filename=TEST_FILE_PATH / 'example_msd2.hdf5')['_dg']
        diff_exc = Diffusion(dg=msd_load)
        diff_exc.dg['da'] = diff_exc.dg['da'] * sc.scalar(1.0, unit=sc.Unit('watts'))

    def test_posterior_predictive_error(self):
        diff_exc = Diffusion(dg=self.msd_load)
        start_dt = 400 * sc.Unit('femtosecond')
        diff_exc._diffusion(start_dt)
        diff_exc._covariance_matrix = diff_exc._covariance_matrix * sc.scalar(1.0, unit=sc.Unit('angstrom6'))

        with pytest.raises(ValueError, match='Units of the covariance matrix and mu do not align correctly'):
            diff_exc.posterior_predictive(n_posterior_samples=1, n_predictive_samples=400)


class TestConsistencyCheck(unittest.TestCase):
    """
    Testing the covariance consistency check.
    """

    @staticmethod
    def _datagroup(gradient=6.0, n_points=64, seed=0):
        """
        A DataGroup with a straight-line mean-squared displacement and a
        realistic number of independent samples at each lag time.
        """
        rng = np.random.RandomState(seed)
        dt = np.arange(2, 2 + n_points, dtype=float)
        n_samples = 4096.0 / dt
        variances = 24.0 * dt**2 / n_samples
        values = gradient * dt + rng.normal(0, np.sqrt(variances) * 0.05)
        da = sc.DataArray(
            sc.array(dims=['time interval'], values=values, variances=variances, unit='angstrom**2'),
            coords={
                'time interval': sc.array(dims=['time interval'], values=dt, unit='ps'),
                'n_samples': sc.array(dims=['time interval'], values=n_samples),
            },
        )
        return sc.DataGroup({'da': da, 'dimensionality': sc.scalar(3)})

    def _fit(self, **kwargs):
        diff = Diffusion(self._datagroup(**kwargs))
        diff.bayesian_regression(sc.scalar(2.0, unit='ps'), progress=False, n_samples=200, n_burn=100)
        return diff

    def test_ratio_is_near_unity_for_a_healthy_fit(self):
        diff = self._fit()
        assert 0.8 < diff.consistency_ratio < 1.25

    def test_no_warning_for_a_healthy_fit(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._fit(seed=1)
        assert not [w for w in caught if 'ordinary least squares' in str(w.message)]

    def test_warning_when_the_gradient_is_suppressed(self):
        diff = self._fit()
        diff.gradient = Samples(diff.gradient.values * 0.01, unit=diff.gradient.unit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            diff._check_consistency()
        assert any('ordinary least squares' in str(w.message) for w in caught)
        assert diff.consistency_ratio < 0.7

    def test_threshold_is_respected(self):
        diff = self._fit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            diff._check_consistency(warn_threshold=1.5)
        assert any('ordinary least squares' in str(w.message) for w in caught)

    def test_check_can_be_disabled(self):
        diff = Diffusion(self._datagroup())
        diff.bayesian_regression(
            sc.scalar(2.0, unit='ps'), progress=False, n_samples=200, n_burn=100, check_consistency=False
        )
        assert diff.consistency_ratio is None

    def test_check_is_safe_before_a_regression(self):
        diff = Diffusion(self._datagroup())
        diff._check_consistency()
        assert diff.consistency_ratio is None
