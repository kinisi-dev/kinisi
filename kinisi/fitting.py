"""
Base classes for fitting analysis with MCMC sampling and Bayesian evidence estimation.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61), Fabian Zills (pythonfz)

from collections.abc import Callable

import numpy as np
import scipp as sc
from dynesty import NestedSampler
from emcee import EnsembleSampler
from scipy.linalg import pinvh
from scipy.optimize import minimize
from scipy.stats import uniform

from kinisi.samples import Samples


class FittingBase:
    """
    A base class for fitting analysis with MCMC sampling and Bayesian evidence estimation.

    This class provides common functionality for fitting models to data with uncertainties,
    including maximum likelihood estimation, MCMC sampling, and nested sampling for
    Bayesian evidence calculation.

    :param data: Data to fit (sc.DataArray with variances)
    :param function: A callable function that describes the relationship
    :param parameter_names: A tuple of parameter names for the function
    :param parameter_units: A tuple of sc.Unit objects corresponding to the parameter names
    :param priors: Prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the max likelihood fit values.
    :param coordinate_name: Name of the coordinate to use as independent variable
    """

    def __init__(
        self,
        data,
        function: Callable,
        parameter_names: tuple[str, ...],
        parameter_units: tuple[sc.Unit, ...],
        priors: None | list = None,
        coordinate_name: str | None = None,
    ) -> 'FittingBase':
        self.data = data
        self.data_group = sc.DataGroup({'data': data})
        self.function = function
        self.priors = priors
        self.parameter_names = parameter_names
        self.parameter_units = parameter_units
        self.coordinate_name = coordinate_name

        if priors is not None:
            if len(priors) != len(self.parameter_names):
                raise ValueError(
                    f'Priors must be a list of length {len(self.parameter_names)}, got {len(priors)} instead.'
                )
            if not all(hasattr(p, 'logpdf') and hasattr(p, 'ppf') for p in priors):
                raise ValueError(
                    "Priors must provide 'logpdf' and 'ppf' methods (e.g., frozen scipy.stats distributions)."
                )

        if self.priors is None:
            # Perform initial fit
            self.max_likelihood()
        else:
            self.max_aposteriori()

        # Set default priors if not provided
        if self.priors is None:
            bounds = tuple(
                [
                    (
                        self.data_group[p].value * 0.5 * self.data_group[p].unit,
                        self.data_group[p].value * 1.5 * self.data_group[p].unit,
                    )
                    for p in self.parameter_names
                ]
            )

            # Set up priors for nested sampling
            self.priors = [uniform(b[0].value, b[1].value - b[0].value) for b in bounds]

    def __repr__(self):
        """String representation."""
        return self.data_group.__repr__()

    def __str__(self):
        """String representation."""
        return self.data_group.__str__()

    def _repr_html_(self):
        """HTML representation."""
        return self.data_group._repr_html_()

    @property
    def logz(self) -> sc.Variable:
        """The log evidence of the model."""
        return self.data_group['logz']

    def get_independent_variable(self):
        """Get the independent variable values for fitting."""
        if self.coordinate_name:
            return self.data.coords[self.coordinate_name].values
        else:
            # Default to first coordinate
            coord_name = list(self.data.coords.keys())[0]
            return self.data.coords[coord_name].values

    def log_likelihood(self, parameters: tuple[float]) -> float:
        """
        Calculate the likelihood of the model given the data.

        :param parameters: The parameters of the model.
        :return: The likelihood of the model.
        """
        x_values = self.get_independent_variable()
        model = self.function(x_values, *parameters)

        covariance_matrix = np.diag(self.data.variances)
        y_values = self.data.values

        _, logdet = np.linalg.slogdet(covariance_matrix)
        logdet += np.log(2 * np.pi) * y_values.size
        inv = pinvh(covariance_matrix)

        diff = model - y_values
        logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))

        return logl

    def nll(self, parameters: tuple[float]) -> float:
        """
        Calculate the negative log likelihood of the model given the data.

        :param parameters: The parameters of the model.
        :return: The negative log likelihood of the model.
        """
        return -self.log_likelihood(parameters)

    def log_prior(self, parameters: tuple[float]) -> float:
        """
        Calculate the log prior probability of the model parameters using a uniform prior.

        :param parameters: The parameters of the model.
        :return: The log prior probability of the model parameters.
        """
        return np.sum([self.priors[i].logpdf(parameters[i]) for i in range(len(parameters))])

    def log_posterior(self, parameters: tuple[float]) -> float:
        """
        Calculate the log posterior probability of the model parameters.

        :param parameters: The parameters of the model.
        :return: The log posterior probability of the model parameters.
        """
        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def nlp(self, parameters: tuple[float]) -> float:
        """
        Calculate the negative log posterior of the model given the data.

        :param parameters: The parameters of the model.
        :return: The negative log posterior of the model.
        """
        return -self.log_posterior(parameters)

    def prior_transform(self, parameters: tuple[float]) -> tuple[float]:
        """
        Transform the parameters from the prior space to the parameter space.

        :param parameters: The parameters of the model in prior space.
        :return: The parameters of the model in parameter space.
        """
        x = np.array(parameters)
        for i in range(len(x)):
            x[i] = self.priors[i].ppf(parameters[i])
        return x

    def max_likelihood(self) -> None:
        """Find the max likelihood fit parameters for the model."""
        if self.priors is not None:
            x0 = [p.mean() for p in self.priors]
        else:
            x0 = [1 for u in self.parameter_units]
        result = minimize(self.nll, x0).x
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = result[i] * self.parameter_units[i]

    def max_aposteriori(self) -> None:
        """Find the max aposteriori fit parameters for the model."""
        if self.priors is not None:
            x0 = [p.mean() for p in self.priors]
        else:
            x0 = [1 for u in self.parameter_units]
        result = minimize(self.nlp, x0).x
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = result[i] * self.parameter_units[i]

    def mcmc(self, n_samples: int = 1000, n_walkers: int = 32, n_burn: int = 500, n_thin=10) -> None:
        """
        Perform MCMC sampling of the model parameters.

        :param n_samples: Number of samples to generate
        :param n_walkers: Number of MCMC walkers
        :param n_burn: Number of burn-in samples
        :param n_thin: Thinning factor
        """
        if isinstance(self.data_group[self.parameter_names[0]], Samples):
            values = np.array([sc.mean(self.data_group[p]).value for p in self.parameter_names])
        else:
            values = np.array([self.data_group[p].value for p in self.parameter_names])
        pos = values + values * 1e-2 * np.random.randn(n_walkers, len(self.parameter_names))
        nwalkers, ndim = pos.shape

        sampler = EnsembleSampler(nwalkers, ndim, self.log_posterior)
        sampler.run_mcmc(pos, n_samples + n_burn, progress=True, progress_kwargs={'desc': 'MCMC Sampling'})
        flatchain = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = Samples(flatchain[:, i], unit=self.parameter_units[i])

    def nested_sampling(self) -> None:
        """Perform nested sampling to estimate the Bayesian evidence of the model."""
        sampler = NestedSampler(self.log_likelihood, self.prior_transform, ndim=len(self.parameter_names))
        sampler.run_nested()
        self.data_group['logz'] = sc.scalar(
            value=sampler.results.logz[-1], variance=sampler.results.logzerr[-1], unit=sc.units.dimensionless
        )
        equal_weighted_samples = sampler.results.samples_equal()
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = Samples(equal_weighted_samples[:, i], unit=self.parameter_units[i])

    @property
    def flatchain(self) -> sc.DataGroup:
        """The flatchain of the MCMC samples."""
        flatchain = {name: self.data_group[name] for name in self.parameter_names}
        return sc.DataGroup(**flatchain)

    @property
    def distribution(self) -> np.ndarray:
        """
        A distribution of samples for the relationship that can be used for easy
        plotting of credible intervals.
        """
        if isinstance(self.data_group[self.parameter_names[0]], Samples):
            parameters = np.array([self.data_group[name].values for name in self.parameter_names])
            x_values = self.get_independent_variable()
            return self.function(x_values[:, np.newaxis], *parameters)
        else:
            raise ValueError(
                'Distribution can only be calculated for Samples objects. Please run mcmc() first to obtain Samples.'
            )
