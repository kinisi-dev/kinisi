"""
Calculate the diffusion coefficient.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Oskar G. Soulas (osoulas)

import warnings

import numpy as np
import scipp as sc
from emcee import EnsembleSampler
from scipp.constants import k
from scipy.linalg import pinvh
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, linregress, multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.stats.moment_helpers import corr2cov, cov2corr
from tqdm import tqdm

from kinisi import __version__
from kinisi.samples import Samples

from .due import Doi, due


class Diffusion:
    """
    The class for the calculation of the self-diffusion coefficient.

    :param dg: A :py:class:`scipp.DataGroup` object containing the relevant mean-squared displacement
        data and number of independent samples.
    """

    def __init__(self, dg: sc.DataGroup):
        self.dg = dg
        self.gradient = None
        self.intercept = None
        self._diffusion_coefficient = None
        self._jump_diffusion_coefficient = None
        self._start_dt = None
        self._cond_max = None
        self._covariance_matrix = None

    def _to_datagroup(self) -> sc.DataGroup:
        """
        Convert the :py:class:`Diffusion` object to a :py:mod:`scipp` DataGroup.
        :return: A :py:mod:`scipp` DataGroup representing the :py:class:`Diffusion` object.
        """
        group = self.__dict__.copy()
        for key, value in group.items():
            if isinstance(value, Samples):
                group[key] = value._to_datagroup()
            elif value is None:
                group[key] = sc.scalar(value=np.nan, dtype='float64')
        return sc.DataGroup(group)

    @classmethod
    def _from_datagroup(cls, datagroup) -> 'Diffusion':
        """
        Convert a :py:mod:`scipp` DataGroup back to a :py:class:`Diffusion` object.
        :return: A :py:class:`Diffusion` object.
        """
        obj = cls.__new__(cls)

        for key, value in datagroup.items():
            if isinstance(value, sc.Variable) and value.ndim == 0 and np.isnan(value.value):
                setattr(obj, key, None)
            elif (
                isinstance(value, sc.DataGroup)
                and '__class__' in value.keys()
                and value['__class__'] == 'kinisi.samples.Samples'
            ):
                setattr(obj, key, Samples._from_datagroup(value))
            else:
                setattr(obj, key, value)

        return obj

    @property
    def covariance_matrix(self) -> sc.Variable:
        """
        :return: The covariance matrix as a :py:mod:`scipp` object, with dimensions of `time_interval1` and
            `time_interval2`.
        """
        return self._covariance_matrix

    @due.dcite(
        Doi('10.1021/acs.jctc.4c01249'),
        path='kinisi.diffusion.Diffusion.bayesian_regression',
        description='Performs the Bayesian regression approach.',
        version=__version__,
    )
    def bayesian_regression(
        self,
        start_dt: sc.Variable,
        cond_max: float = 1e16,
        recondition: bool = False,
        check_consistency: bool = True,
        z_threshold: float = 3.0,
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
        :param recondition: Whether to recondition the covariance matrix. Optional, default is :py:attr:`False`.
        :param check_consistency: Whether to compare the generalised least squares gradient against an ordinary least squares estimate and warn if they disagree by more than the noise in the data allows, if the covariance matrix is too close to singular for the comparison, or if the data show no increase. Optional, default is :py:attr:`True`.
        :param z_threshold: The magnitude of the consistency statistic, :py:attr:`consistency_z`, above which a warning is raised. Optional, default is :py:attr:`3.0`.
        :param fit_intercept: Whether to fit an intercept. Optional, default is :py:attr:`True`.
        :param n_samples: The number of MCMC samples to take. Optional, default is :py:attr:`1000`.
        :param n_walkers: The number of walkers to use in the MCMC. Optional, default is :py:attr:`32`.
        :param n_burn: The number of burn-in samples to discard. Optional, default is :py:attr:`500`.
        :param n_thin: The thinning factor for the MCMC samples. Optional, default is :py:attr:`10`.
        :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.
        :param random_state: The random state to use for the MCMC. Optional, default is :py:attr:`None`.

        :raises ValueError: If :py:attr:`start_dt` leaves fewer than two time intervals, or the mean-squared
            displacements from :py:attr:`start_dt` onwards are not all finite.
        """
        if random_state is not None:
            np.random.seed(random_state.get_state()[1][1])

        self._start_dt = sc.to_unit(start_dt, self.dg['da'].coords['time interval'].unit)
        self._cond_max = cond_max
        self._recondition = recondition
        self._consistency_ratio = None
        self._consistency_z = None

        self.diff_regime = self._diff_regime_index()
        self._covariance_matrix = self.compute_covariance_matrix()

        x_values = self.dg['da'].coords['time interval'][self.diff_regime :].values
        y_values = self.dg['da']['time interval', self.diff_regime :].values
        if x_values.size < 2:
            raise ValueError(
                f'start_dt ({self._start_dt.value} {self._start_dt.unit}) leaves only one time interval, but at '
                'least two are needed for the fit.'
            )
        if not np.all(np.isfinite(y_values)):
            raise ValueError('The mean-squared displacements from start_dt onwards are not all finite.')

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
        self._ols_gradient = ols.slope

        if check_consistency:
            if not ols.slope > 0:
                warnings.warn(
                    f'The ordinary least squares gradient is {ols.slope:.3e}, which is not positive. The data '
                    'show no increase over the fitted time intervals, and the diffusion coefficient may be '
                    'unreliable. Consider whether start_dt lies within the diffusive regime.',
                    UserWarning,
                    stacklevel=2,
                )
            gls_gradient, ols_gradient, gls_variance, ols_variance = _gradient_estimates(
                x_values, y_values, self._covariance_matrix.values, fit_intercept
            )
            self._consistency_z = _check_consistency(
                gls_gradient, ols_gradient, gls_variance, ols_variance, z_threshold
            )

        slope = ols.slope
        intercept = 1e-20
        if slope < 0:
            slope = 1e-20

        if fit_intercept:
            max_likelihood = np.array([slope, intercept])
        else:
            max_likelihood = np.array([slope])

        pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(n_walkers, max_likelihood.size)
        sampler = EnsembleSampler(*pos.shape, log_likelihood)
        # Waiting on https://github.com/dfm/emcee/pull/376
        # if random_state is not None:
        #     pos = max_likelihood + max_likelihood * 1e-3 * random_state.randn(n_walkers, max_likelihood.size)
        #     sampler._random = random_state
        sampler.run_mcmc(pos, n_samples + n_burn, progress=progress, progress_kwargs={'desc': 'Likelihood Sampling'})
        self._flatchain = sampler.get_chain(flat=True, thin=n_thin, discard=n_burn)
        self.gradient = Samples(
            self._flatchain[:, 0], unit=(self.dg['da'].unit / self.dg['da'].coords['time interval'].unit)
        )

        if fit_intercept:
            self.intercept = Samples(self._flatchain[:, 1], unit=self.dg['da'].unit)
        else:
            self.intercept = None

        if check_consistency and ols_gradient > 0:
            self._consistency_ratio = np.median(self.gradient.values) / ols_gradient

    def _diff_regime_index(self) -> int:
        """
        Find the index of the first time interval in the diffusive regime.

        :raises ValueError: If :py:attr:`start_dt` is later than the longest time interval.

        :return: The index of the first time interval that is at least :py:attr:`start_dt`.
        """
        time_interval = self.dg['da'].coords['time interval']
        indices = np.argwhere(time_interval >= self._start_dt)
        if indices.size == 0:
            raise ValueError(
                f'start_dt ({self._start_dt.value} {self._start_dt.unit}) is later than the longest time interval '
                f'({time_interval.max().value} {time_interval.unit}).'
            )
        return indices[0][0]

    @property
    def consistency_ratio(self) -> float:
        """
        The ratio of the median fitted gradient to an ordinary least squares estimate of the same straight
        line, with or without an intercept as fitted.

        This is for information only: how far the ratio can stray from unity by chance depends on the
        noise in the data, so the warning raised by the consistency check uses :py:attr:`consistency_z`
        instead. :py:attr:`None` if the check has not been run or the ordinary least squares gradient is
        not positive.

        :return: The ratio of the two gradient estimates.
        """
        return getattr(self, '_consistency_ratio', None)

    @property
    def consistency_z(self) -> float:
        """
        The difference between the generalised and ordinary least squares gradients of the fitted data,
        divided by the standard deviation of that difference expected from the covariance matrix.

        If the covariance matrix is correct and a straight line describes the data, this follows approximately
        a standard normal distribution, so a magnitude well above three indicates that the covariance matrix
        is unreliable, that the fitted time intervals extend outside the diffusive regime, or that an
        intercept is needed but not fitted. Zero if the two estimators coincide (for example, for a straight
        line through two points), :py:attr:`nan` if the covariance matrix is too close to singular for the
        statistic to be computed, and :py:attr:`None` if the check has not been run.

        :return: The consistency statistic.
        """
        return getattr(self, '_consistency_z', None)

    def _diffusion(self, start_dt: sc.Variable, **kwargs):
        """
        Calculation of the diffusion coefficient.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """
        self.bayesian_regression(start_dt=start_dt, **kwargs)
        _diffusion_coefficient = sc.to_unit(self.gradient / (2 * self.dg['dimensionality'].value), 'cm2/s')
        self._diffusion_coefficient = Samples(_diffusion_coefficient.values, _diffusion_coefficient.unit)

    def _jump_diffusion(self, start_dt: sc.Variable, **kwargs):
        """
        Calculation of the jump diffusion coefficient.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """

        self.bayesian_regression(start_dt=start_dt, **kwargs)
        _jump_diffusion_coefficient = sc.to_unit(self.gradient / (2 * self.dg['dimensionality'].value), 'cm2/s')
        self._jump_diffusion_coefficient = Samples(_jump_diffusion_coefficient.values, _jump_diffusion_coefficient.unit)

    def _conductivity(
        self, start_dt: sc.Variable, temperature: sc.Variable, volume: sc.Variable, number_of_particles: int, **kwargs
    ):
        """
        Calculation of the conductivity.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method.

        :param start_dt: The time at which the diffusion regime begins.
        :param temperature: The temperature of the system.
        :param volume: The volume of the system.
        :param number_of_particles: The number of particles in the system.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """
        self.bayesian_regression(start_dt=start_dt, **kwargs)
        _conductivity_diffusion_coefficient = sc.to_unit(
            self.gradient / (2 * self.dg['dimensionality'].value), 'coulomb2cm2/s'
        )
        volume = sc.sum(volume)
        conversion = number_of_particles / (volume * k * temperature)
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
        :return: The conductivity as a :py:mod:`scipp` object, or :py:attr:`None` if it has not been calculated.
        """
        return getattr(self, '_sigma', None)

    def compute_covariance_matrix(self) -> sc.Variable:
        """
        Compute the covariance matrix for the diffusion coefficient calculation.

        :returns: A :py:mod:`scipp` object containing the covariance matrix.
        """
        cov = np.zeros((self.dg['da'].data.variances.size, self.dg['da'].data.variances.size))
        for i in range(0, self.dg['da'].data.variances.size):
            for j in range(i, self.dg['da'].data.variances.size):
                ratio = self.dg['da'].coords['n_samples'].values[i] / self.dg['da'].coords['n_samples'].values[j]
                value = ratio * self.dg['da'].data.variances[i]
                cov[i, j] = value
                cov[j, i] = np.copy(cov[i, j])
        if self._recondition:
            return sc.array(
                dims=['time_interval1', 'time_interval2'],
                values=cov_nearest(eigenvalue_clipping(cov_nearest(cov[self.diff_regime :, self.diff_regime :]))),
                unit=self.dg['da'].unit ** 2,
            )
        else:
            return sc.array(
                dims=['time_interval1', 'time_interval2'],
                values=cov_nearest(
                    minimum_eigenvalue_method(cov[self.diff_regime :, self.diff_regime :], self._cond_max)
                ),
                unit=self.dg['da'].unit ** 2,
            )

    def posterior_predictive(
        self, n_posterior_samples: int = None, n_predictive_samples: int = 256, progress: bool = True
    ) -> sc.Variable:
        """
        Sample the posterior predictive distribution of the mean-squared displacement in the diffusive regime.
        The result has dimensions of `samples` (of length `n_posterior_samples * n_predictive_samples`) and
        `time interval` (the time intervals from `start_dt` onwards).

        :param n_posterior_samples: Number of samples from the posterior distribution.
            Optional, default is the number of posterior samples.
        :param n_predictive_samples: Number of random samples per sample from the posterior distribution.
            Optional, default is :py:attr:`256`.
        :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.

        :return: Samples from the posterior predictive distribution.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.gradient.size

        ppd_unit = self.gradient.unit * self.dg['da'].coords['time interval'].unit

        diff_regime = self._diff_regime_index()
        ppd = sc.zeros(
            dims=['posterior samples', 'predictive samples', 'time interval'],
            shape=[n_posterior_samples, n_predictive_samples, self.dg['da'].coords['time interval'][diff_regime:].size],
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
            mu = self.gradient[n] * self.dg['da'].coords['time interval'][diff_regime:]
            if self.intercept is not None:
                mu += self.intercept[n]
            mv = multivariate_normal(mean=mu.values, cov=self._covariance_matrix.values, allow_singular=True)
            ppd.values[i] = mv.rvs(n_predictive_samples)

        ppd = sc.flatten(ppd, dims=['posterior samples', 'predictive samples'], to='samples')
        return ppd


@due.dcite(
    Doi('10.1080/16000870.2019.1696646'),
    path='kinisi.diffusion.minimum_eigenvalue_method',
    description='Implements the minimum eigenvalue method for matrix reconditioning.',
    version=__version__,
)
def minimum_eigenvalue_method(cov: np.ndarray, cond_max=1e16) -> np.ndarray:
    """
    Implementation of the matrix reconditioning method known as the minimum
    eigenvalue method, as outlined in doi:10.1080/16000870.2019.1696646. This
    should produce a matrix with a condition number of :py:attr:`cond_max` based
    on the eigenvalues and eigenvectors of the input matrix.

    :param cov: Symmetric matrix to recondition.
    :param cond_max: Expected condition number of output matrix. Optional,
        default is :py:attr:`1e16`.

    :return: Reconditioned matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    T = eigenvalues.max() / cond_max
    new_eigenvalues = np.maximum(eigenvalues, T)
    new_cov = np.real(eigenvectors @ np.diag(new_eigenvalues) @ eigenvectors.T)
    return new_cov


def eigenvalue_clipping(cov: np.ndarray) -> np.ndarray:
    """
    Eigenvalue clipping method for matrix reconditioning.

    :param cov: Covariance matrix to recondition.

    :return: Reconditioned covariance matrix.
    """
    corr = cov2corr(cov)
    eigenthings = np.linalg.eig(corr)
    eigenvalues = eigenthings.eigenvalues.real

    kde = gaussian_kde(eigenvalues)
    x = np.linspace(eigenvalues.min() - 0.5 * eigenvalues.max(), eigenvalues.max() + 0.5 * eigenvalues.max(), 10000)

    popt, _ = curve_fit(marchenkopastur, x, kde.pdf(x), bounds=([0, 0], [np.inf, np.inf]), p0=[0.5, 1.0])

    lambda_plus = (1 + popt[0] ** 0.5) ** 2
    lambda_minus = (1 - popt[0] ** 0.5) ** 2

    lambda_minus = np.max(eigenvalues[eigenvalues < lambda_plus])
    new_eigenvalues = np.copy(eigenvalues)
    new_eigenvalues[new_eigenvalues < lambda_plus] = lambda_minus

    new_corr = (eigenthings.eigenvectors @ np.diag(new_eigenvalues) @ np.linalg.inv(eigenthings.eigenvectors)).real
    S = np.diag(1 / (np.diag(new_corr)) ** 0.5)
    new_corr = S @ new_corr @ S.T

    new_cov = corr2cov(new_corr, np.sqrt(cov.diagonal()))
    return new_cov


def marchenkopastur(x: np.ndarray, lambda_: float, sigma: float) -> np.ndarray:
    """
    Marchenko-Pastur distribution

    :param x: points at which to evaluate the distribution
    :param lambda_: lambda parameter
    :param sigma: standard deviation of the distribution
    """

    def m0(a: np.ndarray) -> np.ndarray:
        """
        Element wise maximum of (a, 0)

        :param a: input array
        :return: element wise maximum of (a, 0)
        """
        return np.maximum(a, np.zeros_like(a))

    lambda_plus = (1 + lambda_**0.5) ** 2
    lambda_minus = (1 - lambda_**0.5) ** 2

    return np.sqrt(m0(lambda_plus - x) * m0(x - lambda_minus)) / (2 * np.pi * sigma**2 * lambda_ * x)


def _straight_line(abscissa: np.ndarray, gradient: float, intercept: float = 0.0) -> np.ndarray:
    """
    A one dimensional straight line function.

    :param abscissa: The abscissa data.
    :param gradient: The slope of the line.
    :param intercept: The y-intercept of the line. Optional, default is :py:attr:`0.0`.

    :return: The resulting ordinate.
    """
    return gradient * abscissa + intercept


def _gradient_estimates(
    abscissa: np.ndarray, ordinate: np.ndarray, covariance: np.ndarray, fit_intercept: bool = True
) -> tuple[float, float, float, float]:
    """
    Estimate the gradient of a straight line by generalised and by ordinary least squares, together
    with the variance of each estimate under the given covariance matrix.

    The generalised least squares estimate uses the same pseudo-inverse of the covariance matrix as
    the likelihood in :py:func:`Diffusion.bayesian_regression`. If the normal matrix formed with that
    pseudo-inverse is exactly singular (in practice, only when every eigenvalue of the covariance matrix
    is discarded), the generalised least squares gradient is :py:attr:`nan` and its variance is infinite.
    A normal matrix that is nearly singular instead gives a very large or negative variance, which
    :py:func:`_check_consistency` detects.

    :param abscissa: The abscissa data.
    :param ordinate: The ordinate data.
    :param covariance: The covariance matrix of the ordinate data.
    :param fit_intercept: Whether the straight line has an intercept. Optional, default is :py:attr:`True`.

    :return: The generalised least squares gradient, the ordinary least squares gradient, and the
        variances of the generalised and ordinary least squares gradients.
    """
    if fit_intercept:
        design = np.column_stack([abscissa, np.ones_like(abscissa)])
    else:
        design = abscissa[:, np.newaxis]

    ols_inverse = np.linalg.inv(design.T @ design)
    ols_gradient = (ols_inverse @ design.T @ ordinate)[0]
    ols_variance = (ols_inverse @ design.T @ covariance @ design @ ols_inverse)[0, 0]

    weights = pinvh(covariance)
    try:
        gls_inverse = np.linalg.inv(design.T @ weights @ design)
    except np.linalg.LinAlgError:
        return np.nan, ols_gradient, np.inf, ols_variance
    gls_gradient = (gls_inverse @ design.T @ weights @ ordinate)[0]
    gls_variance = gls_inverse[0, 0]

    return gls_gradient, ols_gradient, gls_variance, ols_variance


def _check_consistency(
    gls_gradient: float, ols_gradient: float, gls_variance: float, ols_variance: float, z_threshold: float = 3.0
) -> float:
    """
    Compare the generalised least squares gradient with the ordinary least squares gradient of the same
    data, and warn if they disagree by more than the noise in the data allows.

    The generalised least squares fit performed by :py:func:`Diffusion.bayesian_regression` weights the
    observations by the inverse of an estimated covariance matrix. Where sampling noise has made that
    matrix numerically singular, the inversion can return a gradient in error by several orders of
    magnitude, in either direction, without any other indication of failure. An ordinary least squares
    fit to the same observations does not invert the covariance and so cannot be affected in the same way.

    If the covariance matrix is correct, both estimators are unbiased and the generalised least squares
    estimator has the smaller variance, so the statistic z = (g_GLS - g_OLS) / sqrt(Var_OLS - Var_GLS)
    follows approximately a standard normal distribution (a Hausman test). This is exact only for normally
    distributed data with a known covariance matrix that the pseudo-inverse inverts exactly; a covariance
    matrix estimated from the same data widens the distribution somewhat. A large value of abs(z) is
    evidence that the covariance matrix is unreliable, or that a straight line does not describe the data
    over the fitted time intervals (for example, if these begin before the diffusive regime or an intercept
    is needed but not fitted), rather than that the data are unusual.

    A generalised least squares variance that is not positive, or larger than the ordinary least squares
    variance, cannot arise from an exact inverse of the covariance matrix, and shows that the matrix is too
    close to singular for its pseudo-inverse. If the two variances are equal, to within rounding, the two
    estimators coincide (for example, for a straight line through two points), so they cannot disagree and
    z is taken as zero.

    :param gls_gradient: The generalised least squares gradient.
    :param ols_gradient: The ordinary least squares gradient.
    :param gls_variance: The variance of the generalised least squares gradient.
    :param ols_variance: The variance of the ordinary least squares gradient.
    :param z_threshold: The magnitude of z above which a warning is raised. Optional, default
        is :py:attr:`3.0`.

    :return: The statistic z, or :py:attr:`nan` if the covariance matrix is too close to singular or a
        gradient is not finite.
    """
    advice = 'Consider refitting with recondition=True, or inspecting the eigenvalues of Diffusion.covariance_matrix.'

    # Rounding leaves a relative difference of order 1e-10 between the variances of estimators that coincide.
    difference = ols_variance - gls_variance
    tolerance = 1e-8 * ols_variance

    if not gls_variance > 0 or not difference > -tolerance:
        warnings.warn(
            'The generalised least squares gradient has a larger variance than the ordinary least squares '
            'gradient, or a variance that is not positive, which is impossible for a well-conditioned '
            'covariance matrix. The covariance matrix is too close to singular for its pseudo-inverse to be '
            f'reliable, and the diffusion coefficient may be unreliable. {advice}',
            UserWarning,
            stacklevel=3,
        )
        return np.nan

    if difference <= tolerance:
        return 0.0

    z = (gls_gradient - ols_gradient) / np.sqrt(difference)

    if not np.isfinite(z):
        warnings.warn(
            'The generalised or ordinary least squares gradient is not finite, so the consistency of the fit '
            f'could not be checked. {advice}',
            UserWarning,
            stacklevel=3,
        )
    elif abs(z) > z_threshold:
        warnings.warn(
            f'The generalised least squares gradient differs from the ordinary least squares estimate by '
            f'{abs(z):.1f} times the spread expected from the noise in the data (z = {z:.2f}). A difference '
            'this large indicates that the covariance matrix may be unreliable, or that a straight line does '
            'not describe the data from start_dt onwards, and that the diffusion coefficient may be wrong. '
            'Consider a later start_dt, refitting with recondition=True, or inspecting the eigenvalues of '
            'Diffusion.covariance_matrix.',
            UserWarning,
            stacklevel=3,
        )

    return z
