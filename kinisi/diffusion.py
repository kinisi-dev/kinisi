"""
Calculate the diffusion coefficient.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
from emcee import EnsembleSampler
from scipp.constants import k
from scipy.linalg import pinvh
from scipy.optimize import minimize
from scipy.stats import linregress, multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from tqdm import tqdm

from kinisi.samples import Samples


class Diffusion:
    """
    The class for the calcualtion of the self-diffusion coefficient.

    :param da: A :py:class:`scipp.DataArray` object containing the relevant mean-squared displacement
        data and number of independent samples.
    """

    def __init__(self, da: sc.DataArray):
        self.da = da
        self.gradient = None
        self.intercept = None
        self._diffusion_coefficient = None
        self._jump_diffusion_coefficient = None
        self._start_dt = None
        self._cond_max = None
        self._covariance_matrix = None

    @property
    def covariance_matrix(self) -> sc.Variable:
        """
        :return: The covariance matrix as a :py:mod:`scipp` object, with dimensions of `time_interval1` and
            `time_interval2`.
        """
        return self._covariance_matrix

    def bayesian_regression(
        self,
        start_dt: sc.Variable,
        cond_max: float = 1e16,
        fit_intercept: bool = True,
        n_samples: int = 1000,
        n_walkers: int = 32,
        n_burn: int = 500,
        n_thin: int = 10,
        progress: bool = True,
        random_state: np.random.mtrand.RandomState = None,
    ):
        """
        Perform the Bayesian regression with a linear model against the observed data.

        :param start_dt: The time at which the diffusion regime begins.
        :param cond_max: The maximum condition number of the covariance matrix. Optional, default is :py:attr:`1e16`.
        :param fit_intercept: Whether to fit an intercept. Optional, default is :py:attr:`True`.
        :param n_samples: The number of MCMC samples to take. Optional, default is :py:attr:`1000`.
        :param n_walkers: The number of walkers to use in the MCMC. Optional, default is :py:attr:`32`.
        :param n_burn: The number of burn-in samples to discard. Optional, default is :py:attr:`500`.
        :param n_thin: The thinning factor for the MCMC samples. Optional, default is :py:attr:`10`.
        :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.
        :param random_state: The random state to use for the MCMC. Optional, default is :py:attr:`None`.
        """
        if random_state is not None:
            np.random.seed(random_state.get_state()[1][1])

        self._start_dt = start_dt
        self._cond_max = cond_max

        self.diff_regime = np.argwhere(self.da.coords['time interval'] >= self._start_dt)[0][0]
        self._covariance_matrix = self.compute_covariance_matrix()

        x_values = self.da.coords['time interval'][self.diff_regime :].values
        y_values = self.da['time interval', self.diff_regime :].values

        _, logdet = np.linalg.slogdet(self._covariance_matrix.values)
        logdet += np.log(2 * np.pi) * y_values.size
        inv = pinvh(self._covariance_matrix.values)

        def log_likelihood(theta: np.ndarray) -> float:
            """
            Get the log likelihood for multivariate normal distribution.
            :param theta: Value of the gradient and intercept of the straight line.
            :return: Log-likelihood value.
            """
            if theta[0] < 0:
                return -np.inf
            model = _straight_line(x_values, *theta)
            diff = model - y_values
            logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
            return logl

        ols = linregress(x_values, y_values)
        slope = ols.slope
        intercept = 1e-20
        if slope < 0:
            slope = 1e-20

        def nll(*args) -> float:
            """
            General purpose negative log-likelihood.
            :return: Negative log-likelihood
            """
            return -log_likelihood(*args)

        if fit_intercept:
            max_likelihood = minimize(nll, np.array([slope, intercept])).x
        else:
            max_likelihood = minimize(nll, np.array([slope])).x

        pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(n_walkers, max_likelihood.size)
        sampler = EnsembleSampler(*pos.shape, log_likelihood)
        # Waiting on https://github.com/dfm/emcee/pull/376
        # if random_state is not None:
        #     pos = max_likelihood + max_likelihood * 1e-3 * random_state.randn(n_walkers, max_likelihood.size)
        #     sampler._random = random_state
        sampler.run_mcmc(pos, n_samples + n_burn, progress=progress, progress_kwargs={'desc': 'Likelihood Sampling'})
        self._flatchain = sampler.get_chain(flat=True, thin=n_thin, discard=n_burn)

        self.gradient = Samples(self._flatchain[:, 0], unit=(self.da.unit / self.da.coords['time interval'].unit))

        if fit_intercept:
            self.intercept = Samples(self._flatchain[:, 1], unit=self.da.unit)

    def _diffusion(self, start_dt: sc.Variable, **kwargs):
        """
        Calculation of the diffusion coefficient.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """
        self.bayesian_regression(start_dt=sc.to_unit(start_dt, self.da.coords['time interval'].unit), **kwargs)
        _diffusion_coefficient = sc.to_unit(self.gradient / (2 * self.da.coords['dimensionality'].value), 'cm2/s')
        self._diffusion_coefficient = Samples(_diffusion_coefficient.values, _diffusion_coefficient.unit)

    def _jump_diffusion(self, start_dt: sc.Variable, **kwargs):
        """
        Calculation of the jump diffusion coefficient.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """

        self.bayesian_regression(start_dt=start_dt, **kwargs)
        _jump_diffusion_coefficient = sc.to_unit(self.gradient / (2 * self.da.coords['dimensionality'].value), 'cm2/s')
        self._jump_diffusion_coefficient = Samples(_jump_diffusion_coefficient.values, _jump_diffusion_coefficient.unit)

    def _conductivity(self, start_dt: sc.Variable, temperature: sc.Variable, volume: sc.Variable, **kwargs):
        """
        Calculation of the conductivity.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param temperature: The temperature of the system.
        :param volume: The volume of the system.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """
        self.bayesian_regression(start_dt=start_dt, **kwargs)
        _conductivity_diffusion_coefficient = sc.to_unit(
            self.gradient / (2 * self.da.coords['dimensionality'].value), 'coulomb2cm2/s'
        )
        conversion = 1 / (volume * k * temperature)
        _sigma = sc.to_unit(_conductivity_diffusion_coefficient * conversion, 'mS/cm')
        self._sigma = Samples(_sigma.values, _sigma.unit)

    @property
    def D(self) -> sc.Variable:
        """
        :return: The diffusion coefficient as a :py:mod:`scipp` object.
        """
        return self._diffusion_coefficient

    @property
    def D_J(self) -> sc.Variable:
        """
        :return: The jump diffusion coefficient as a :py:mod:`scipp` object.
        """
        return self._jump_diffusion_coefficient

    @property
    def sigma(self) -> sc.Variable:
        """
        :return: The conductivity as a :py:mod:`scipp` object.
        """
        return self._sigma

    def compute_covariance_matrix(self) -> sc.Variable:
        """
        Compute the covariance matrix for the diffusion coefficient calculation.

        :returns: A :py:mod:`scipp` object containing the covariance matrix.
        """
        cov = np.zeros((self.da.data.variances.size, self.da.data.variances.size))
        for i in range(0, self.da.data.variances.size):
            for j in range(i, self.da.data.variances.size):
                ratio = self.da.coords['n_samples'].values[i] / self.da.coords['n_samples'].values[j]
                value = ratio * self.da.data.variances[i]
                cov[i, j] = value
                cov[j, i] = np.copy(cov[i, j])
        return sc.array(
            dims=['time_interval1', 'time_interval2'],
            values=cov_nearest(minimum_eigenvalue_method(cov[self.diff_regime :, self.diff_regime :], self._cond_max)),
            unit=self.da.unit**2,
        )

    def posterior_predictive(
        self, n_posterior_samples: int = None, n_predictive_samples: int = 256, progress: bool = True
    ) -> sc.Variable:
        """
        Sample the posterior predictive distribution. The shape of the resulting array will be
        `(n_posterior_samples * n_predictive_samples, start_dt)`.

        :param n_posterior_samples: Number of samples from the posterior distribution.
            Optional, default is the number of posterior samples.
        :param n_predictive_samples: Number of random samples per sample from the posterior distribution.
            Optional, default is :py:attr:`256`.
        :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.

        :return: Samples from the posterior predictive distribution.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.gradient.size

        ppd_unit = self.gradient.unit * self.da.coords['time interval'].unit + self.intercept.unit

        diff_regime = np.argwhere(self.da.coords['time interval'] >= self._start_dt)[0][0]
        ppd = sc.zeros(
            dims=['posterior samples', 'predictive samples', 'time interval'],
            shape=[n_posterior_samples, n_predictive_samples, self.da.coords['time interval'][diff_regime:].size],
            unit=ppd_unit,
        )
        samples_to_draw = list(enumerate(np.random.randint(0, self.gradient.size, size=n_posterior_samples)))

        if progress:
            iterator = tqdm(samples_to_draw, desc='Calculating Posterior Predictive')
        else:
            iterator = samples_to_draw

        # Checking unit consistency for mu and covariance
        if ppd_unit**2 != self._covariance_matrix.unit:
            raise ValueError('Units of the covariance matrix and mu do not align correctly.')

        for i, n in iterator:
            mu = self.gradient[n] * self.da.coords['time interval'][diff_regime:] + self.intercept[n]
            mv = multivariate_normal(mean=mu.values, cov=self._covariance_matrix.values, allow_singular=True)
            ppd.values[i] = mv.rvs(n_predictive_samples)

        ppd = sc.flatten(ppd, dims=['posterior samples', 'predictive samples'], to='samples')
        return ppd


def minimum_eigenvalue_method(cov: np.ndarray, cond_max=1e16) -> np.ndarray:
    """
    Implementation of the matrix reconditioning method known as the minimum
    eigenvalue method, as outlined in doi:10.1080/16000870.2019.1696646. This
    should produce a matrix with a condition number of :py:attr:`cond_max` based
    on the eigenvalues and eigenvectors of the input matrix.

    :param matrix: Matrix to recondition.
    :param cond_max: Expected condition number of output matrix. Optional,
        default is :py:attr:`1e16`.

    :return: Reconditioned matrix.
    """
    eigenthings = np.linalg.eig(cov)
    eigenvalues = eigenthings.eigenvalues
    new_eigenvalues = np.copy(eigenvalues)
    T = eigenvalues[0] / cond_max
    new_eigenvalues[np.where(new_eigenvalues < T)] = T
    new_cov = np.real(eigenthings.eigenvectors @ np.diag(new_eigenvalues) @ eigenthings.eigenvectors.T)
    return new_cov


def _straight_line(abscissa: np.ndarray, gradient: float, intercept: float = 0.0) -> np.ndarray:
    """
    A one dimensional straight line function.

    :param abscissa: The abscissa data.
    :param gradient: The slope of the line.
    :param intercept: The y-intercept of the line. Optional, default is :py:attr:`0.0`.

    :return: The resulting ordinate.
    """
    return gradient * abscissa + intercept
