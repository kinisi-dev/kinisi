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
from scipy.stats import linregress

from kinisi.diffusion import (
    Diffusion,
    _check_consistency,
    _gradient_estimates,
    _straight_line,
    eigenvalue_clipping,
    marchenkopastur,
    minimum_eigenvalue_method,
)
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

    def test_minimum_eigenvalue_method_condition_number(self):
        # The largest eigenvalue is not the first, so the threshold must use the maximum.
        matrix = np.diag([1.0, 1e-20, 10.0])
        reconditioned_matrix = minimum_eigenvalue_method(matrix, 1e3)
        assert np.allclose(np.linalg.eigvalsh(reconditioned_matrix), [1e-2, 1.0, 10.0])

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

    def test_start_dt_beyond_data(self):
        diff = Diffusion(dg=self.msd_load)
        assert diff.sigma is None
        with pytest.raises(ValueError, match='later than the longest time interval'):
            diff.bayesian_regression(start_dt=1e9 * sc.Unit('femtosecond'))

    def test_bayesian_regression_without_intercept(self):
        diff = Diffusion(dg=self.msd_load)
        diff.bayesian_regression(start_dt=10 * sc.Unit('femtosecond'), fit_intercept=False, n_samples=100, n_burn=50)

        assert diff.intercept is None
        assert diff._flatchain.shape[1] == 1
        ppd = diff.posterior_predictive(n_posterior_samples=2, n_predictive_samples=3, progress=False)
        assert ppd.sizes['samples'] == 6

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

    def test_start_dt_unit_conversion(self):
        diff = Diffusion(dg=self.msd_load)
        diff._jump_diffusion(0.2 * sc.Unit('ps'), n_samples=100, n_burn=50, progress=False)
        assert diff._start_dt.unit == self.msd_load['da'].coords['time interval'].unit
        expected = np.argwhere(self.msd_load['da'].coords['time interval'] >= 200 * sc.Unit('fs'))[0][0]
        assert diff.diff_regime == expected

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


def _msd_like_covariance(dt):
    """
    A covariance matrix shaped like that of a mean-squared displacement: the variances grow with the
    time interval and neighbouring time intervals are strongly positively correlated.
    """
    return 0.01 * np.minimum.outer(dt, dt) ** 2


class TestGradientEstimates(unittest.TestCase):
    """
    Testing the generalised and ordinary least squares gradient estimates.
    """

    dt = np.arange(2, 12, dtype=float)

    def test_both_gradients_recover_a_straight_line(self):
        gls_gradient, ols_gradient, _, _ = _gradient_estimates(self.dt, 6 * self.dt + 2, _msd_like_covariance(self.dt))
        assert np.isclose(gls_gradient, 6)
        assert np.isclose(ols_gradient, 6)

    def test_both_gradients_recover_a_line_through_the_origin(self):
        gls_gradient, ols_gradient, _, _ = _gradient_estimates(
            self.dt, 6 * self.dt, _msd_like_covariance(self.dt), fit_intercept=False
        )
        assert np.isclose(gls_gradient, 6)
        assert np.isclose(ols_gradient, 6)

    def test_ols_gradient_matches_linregress(self):
        y = 6 * self.dt + np.random.RandomState(0).normal(0, 1, self.dt.size)
        _, ols_gradient, _, _ = _gradient_estimates(self.dt, y, _msd_like_covariance(self.dt))
        assert np.isclose(ols_gradient, linregress(self.dt, y).slope)

    def test_estimators_coincide_for_uncorrelated_equal_variances(self):
        y = 6 * self.dt + np.random.RandomState(0).normal(0, 1, self.dt.size)
        gls_gradient, ols_gradient, gls_variance, ols_variance = _gradient_estimates(
            self.dt, y, 4.0 * np.eye(self.dt.size)
        )
        expected_variance = 4.0 / np.sum((self.dt - self.dt.mean()) ** 2)
        assert np.isclose(gls_gradient, ols_gradient)
        assert np.isclose(gls_variance, expected_variance)
        assert np.isclose(ols_variance, expected_variance)

    def test_estimators_coincide_for_uncorrelated_equal_variances_without_intercept(self):
        y = 6 * self.dt + np.random.RandomState(0).normal(0, 1, self.dt.size)
        gls_gradient, ols_gradient, gls_variance, ols_variance = _gradient_estimates(
            self.dt, y, 4.0 * np.eye(self.dt.size), fit_intercept=False
        )
        expected_variance = 4.0 / np.sum(self.dt**2)
        assert np.isclose(gls_gradient, ols_gradient)
        assert np.isclose(gls_variance, expected_variance)
        assert np.isclose(ols_variance, expected_variance)

    def test_coinciding_estimators_pass_the_check(self):
        y = 6 * self.dt + np.random.RandomState(0).normal(0, 1, self.dt.size)
        z, messages = _consistency_warnings(*_gradient_estimates(self.dt, y, 4.0 * np.eye(self.dt.size)))
        assert z == 0
        assert not messages

    def test_z_is_standard_normal_for_the_true_covariance(self):
        covariance = _msd_like_covariance(self.dt)
        cholesky = np.linalg.cholesky(covariance)
        for fit_intercept in [True, False]:
            with self.subTest(fit_intercept=fit_intercept):
                rng = np.random.RandomState(0)
                z = []
                for _ in range(4000):
                    y = 6 * self.dt + 2 * fit_intercept + cholesky @ rng.normal(size=self.dt.size)
                    estimates = _gradient_estimates(self.dt, y, covariance, fit_intercept)
                    z.append(_check_consistency(*estimates, z_threshold=np.inf))
                assert abs(np.mean(z)) < 0.07
                assert 0.93 < np.std(z) < 1.07

    def test_gls_variance_is_smaller_for_a_well_conditioned_covariance(self):
        _, _, gls_variance, ols_variance = _gradient_estimates(self.dt, 6 * self.dt, _msd_like_covariance(self.dt))
        assert 0 < gls_variance < ols_variance

    def test_gls_variance_is_not_smaller_for_a_near_singular_covariance(self):
        # Collapse the covariance along a direction that overlaps the time intervals, so that the
        # pseudo-inverse discards information the ordinary least squares estimate still uses.
        centred = self.dt - self.dt.mean()
        alternating = (-1.0) ** np.arange(self.dt.size)
        direction = centred / np.linalg.norm(centred) + alternating / np.linalg.norm(alternating)
        direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        projection = np.eye(self.dt.size) - np.outer(direction, direction)
        covariance = projection @ _msd_like_covariance(self.dt) @ projection
        estimates = _gradient_estimates(self.dt, 6 * self.dt, covariance)
        _, _, gls_variance, ols_variance = estimates
        assert 0 < gls_variance < np.inf
        assert ols_variance < gls_variance
        z, messages = _consistency_warnings(*estimates)
        assert np.isnan(z)
        assert any('singular' in m for m in messages)

    def test_gls_variance_is_infinite_when_the_covariance_carries_no_information(self):
        gls_gradient, _, gls_variance, _ = _gradient_estimates(
            self.dt, 6 * self.dt, np.zeros((self.dt.size, self.dt.size))
        )
        assert np.isnan(gls_gradient)
        assert np.isinf(gls_variance)


def _consistency_warnings(*args, **kwargs):
    """
    Run the consistency check and return its value and the messages of the warnings it raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        z = _check_consistency(*args, **kwargs)
    return z, [str(w.message) for w in caught]


class TestCheckConsistency(unittest.TestCase):
    """
    Testing the decision of whether the generalised and ordinary least squares gradients agree.

    The variances are chosen so that the standard deviation of the difference between the two
    gradients is one, so the statistic is the difference itself.
    """

    def test_no_warning_when_the_gradients_agree(self):
        z, messages = _consistency_warnings(6.5, 6.0, 1.0, 2.0)
        assert np.isclose(z, 0.5)
        assert not messages

    def test_warning_when_the_gradient_is_inflated(self):
        z, messages = _consistency_warnings(10.0, 6.0, 1.0, 2.0)
        assert np.isclose(z, 4.0)
        assert any('ordinary least squares' in m for m in messages)

    def test_warning_when_the_gradient_is_suppressed(self):
        z, messages = _consistency_warnings(2.0, 6.0, 1.0, 2.0)
        assert np.isclose(z, -4.0)
        assert any('ordinary least squares' in m for m in messages)

    def test_threshold_is_respected(self):
        _, messages = _consistency_warnings(7.0, 6.0, 1.0, 2.0)
        assert not messages
        _, messages = _consistency_warnings(7.0, 6.0, 1.0, 2.0, z_threshold=0.5)
        assert any('ordinary least squares' in m for m in messages)

    def test_no_warning_for_a_negative_gradient(self):
        # Whether the data increase is checked by the regression, not here.
        z, messages = _consistency_warnings(-1.0, -1.0, 1.0, 2.0)
        assert z == 0
        assert not messages

    def test_warning_when_the_ols_variance_is_smaller(self):
        z, messages = _consistency_warnings(6.0, 6.0, 2.0, 1.0)
        assert np.isnan(z)
        assert any('singular' in m for m in messages)

    def test_no_warning_when_the_variances_are_equal(self):
        z, messages = _consistency_warnings(6.0, 6.0, 1.0, 1.0)
        assert z == 0
        assert not messages

    def test_no_warning_when_the_variances_differ_by_rounding(self):
        z, messages = _consistency_warnings(6.0, 6.0, 1.0, 1.0 - 1e-12)
        assert z == 0
        assert not messages

    def test_warning_when_the_gls_variance_is_not_positive(self):
        z, messages = _consistency_warnings(6.0, 6.0, -5e13, 2.0)
        assert np.isnan(z)
        assert any('singular' in m for m in messages)

    def test_warning_when_the_gls_variance_is_infinite(self):
        z, messages = _consistency_warnings(np.nan, 6.0, np.inf, 2.0)
        assert np.isnan(z)
        assert any('singular' in m for m in messages)

    def test_warning_when_the_statistic_is_not_finite(self):
        z, messages = _consistency_warnings(np.nan, 6.0, 1.0, 2.0)
        assert np.isnan(z)
        assert any('not finite' in m for m in messages)


class TestConsistencyCheck(unittest.TestCase):
    """
    Testing the covariance consistency check.
    """

    @staticmethod
    def _datagroup(gradient=6.0, intercept=0.0, n_points=64, seed=0, noise=0.05):
        """
        A DataGroup with a straight-line mean-squared displacement and a
        realistic number of independent samples at each lag time. The noise
        added is the given fraction of the stated standard deviation.
        """
        rng = np.random.RandomState(seed)
        dt = np.arange(2, 2 + n_points, dtype=float)
        n_samples = 4096.0 / dt
        variances = 24.0 * dt**2 / n_samples
        values = gradient * dt + intercept + rng.normal(0, np.sqrt(variances) * noise)
        da = sc.DataArray(
            sc.array(dims=['time interval'], values=values, variances=variances, unit='angstrom**2'),
            coords={
                'time interval': sc.array(dims=['time interval'], values=dt, unit='ps'),
                'n_samples': sc.array(dims=['time interval'], values=n_samples),
            },
        )
        return sc.DataGroup({'da': da, 'dimensionality': sc.scalar(3)})

    @staticmethod
    def _regress(diff, start_dt=2.0, **kwargs):
        diff.bayesian_regression(sc.scalar(start_dt, unit='ps'), progress=False, n_samples=200, n_burn=100, **kwargs)
        return diff

    def _fit(self, **kwargs):
        return self._regress(Diffusion(self._datagroup(**kwargs)))

    def _messages(self, diff, **kwargs):
        """
        Run the regression and return the messages of the warnings it raised.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._regress(diff, **kwargs)
        return [str(w.message) for w in caught]

    @staticmethod
    def _overconfident(diff, factor=1e-3):
        """
        Make the covariance matrix used by the regression too small by the given factor.
        """
        compute_covariance_matrix = diff.compute_covariance_matrix
        diff.compute_covariance_matrix = lambda: compute_covariance_matrix() * factor
        return diff

    def test_ratio_is_near_unity_for_a_healthy_fit(self):
        diff = self._fit()
        assert 0.8 < diff.consistency_ratio < 1.25

    def test_no_warning_for_a_healthy_fit(self):
        messages = self._messages(Diffusion(self._datagroup(seed=1, noise=1.0)))
        assert not [m for m in messages if 'ordinary least squares' in m]

    def test_no_warning_for_a_healthy_fit_without_intercept(self):
        messages = self._messages(Diffusion(self._datagroup(seed=1, noise=1.0)), fit_intercept=False)
        assert not [m for m in messages if 'ordinary least squares' in m]

    def test_z_is_reported_for_a_healthy_fit(self):
        diff = self._fit(noise=1.0)
        assert np.isfinite(diff.consistency_z)
        assert abs(diff.consistency_z) < 3

    def test_warning_for_an_overconfident_covariance(self):
        diff = self._overconfident(Diffusion(self._datagroup(noise=1.0)))
        messages = self._messages(diff)
        assert abs(diff.consistency_z) > 3
        assert any('differs from the ordinary least squares' in m for m in messages)

    def test_z_threshold_is_respected(self):
        diff = self._overconfident(Diffusion(self._datagroup(noise=1.0)))
        assert any('differs from the ordinary least squares' in m for m in self._messages(diff))
        messages = self._messages(diff, z_threshold=abs(diff.consistency_z) + 1)
        assert not [m for m in messages if 'differs from the ordinary least squares' in m]

    def test_warning_when_the_data_do_not_increase(self):
        diff = Diffusion(self._datagroup(gradient=-0.5))
        messages = self._messages(diff)
        assert any('not positive' in m for m in messages)
        assert np.isfinite(diff.consistency_z)
        assert diff.consistency_ratio is None

    def test_warning_when_the_data_do_not_increase_without_intercept(self):
        # A line through the origin has a positive gradient for these data, although they fall.
        diff = Diffusion(self._datagroup(gradient=-0.5, intercept=50.0))
        messages = self._messages(diff, fit_intercept=False)
        assert any('not positive' in m for m in messages)

    def test_estimators_coincide_for_two_time_intervals(self):
        diff = Diffusion(self._datagroup())
        messages = self._messages(diff, start_dt=64.0)
        assert diff.consistency_z == 0
        assert not [m for m in messages if 'singular' in m]

    def test_one_time_interval_is_too_few(self):
        with pytest.raises(ValueError, match='only one time interval'):
            self._regress(Diffusion(self._datagroup()), start_dt=65.0)

    def test_non_finite_data_are_rejected(self):
        dg = self._datagroup()
        dg['da'].values[10] = np.nan
        with pytest.raises(ValueError, match='not all finite'):
            self._regress(Diffusion(dg))

    def test_check_can_be_disabled(self):
        diff = self._regress(Diffusion(self._datagroup()), check_consistency=False)
        assert diff.consistency_ratio is None
        assert diff.consistency_z is None

    def test_disabling_the_check_clears_earlier_results(self):
        diff = self._fit()
        self._regress(diff, check_consistency=False)
        assert diff.consistency_ratio is None
        assert diff.consistency_z is None

    def test_results_are_none_before_a_regression(self):
        diff = Diffusion(self._datagroup())
        assert diff.consistency_ratio is None
        assert diff.consistency_z is None
