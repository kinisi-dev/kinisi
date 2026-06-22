"""
Implementation of the Yeh-Hummer finite-size correction for diffusion coefficients
from molecular dynamics simulations with periodic boundary conditions.

Based on: Yeh & Hummer, J. Phys. Chem. B 2004, 108, 15873-15879
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Fabian Zills (pythonfz)

import numpy as np
import scipp as sc
import scipp.constants as const
from scipy.optimize import curve_fit, minimize
from scipy.stats import uniform

from kinisi import __version__
from kinisi.fitting import FittingBase
from kinisi.samples import Samples

from .due import Doi, due


@due.dcite(
    Doi('10.1021/jp0477147'),
    path='kinisi.yeh_hummer.YehHummer',
    description='Yeh-Hummer finite-size correction.',
    version=__version__,
)
class YehHummer(FittingBase):
    """
    Apply Yeh-Hummer finite-size corrections to diffusion coefficients from MD simulations
    with periodic boundary conditions.

    The Yeh-Hummer correction formula is:
    D_PBC = D_0 - (k_B * T * xi) / (6 * pi * eta * L)

    Internally, fitting is performed in (D_0, slope) space where the model is linear,
    then slope is converted to viscosity for output.

    :param diffusion: sc.DataArray with diffusion coefficients and box_length coordinate
    :param temperature: Temperature (will be extracted from coords if not provided)
    :param priors: Optional priors for [D_0, viscosity] parameters using scipy.stats objects,
        where viscosity has units Pa*s.
    """

    def __init__(self, diffusion, temperature: sc.Variable, priors=None):
        self.diffusion = diffusion

        # Extract box lengths from coordinates
        self.box_lengths = diffusion.coords['box_length']
        self.temperature = temperature

        # Constants
        self.xi_cubic = 2.837297  # Ewald constant for cubic boxes

        # Slope has units of [diffusion] * [length] = cm^2/s * Å
        self._slope_unit = diffusion.unit * self.box_lengths.unit

        # Set up parameters for fitting - use (D_0, slope) for linear model
        parameter_names = ('D_0', 'slope')
        parameter_units = (diffusion.unit, self._slope_unit)

        # Compute priors: use provided or defaults
        if priors is None:
            D_max = np.max(diffusion.values)
            D_prior = uniform(D_max * 0.8, D_max * 1.2)
            visc_lower, visc_upper = 1e-5 * sc.Unit('Pa*s'), 1e-1 * sc.Unit('Pa*s')

            # Higher viscosity = lower slope, so bounds are inverted
            slope_bounds = (
                self.viscosity_to_slope(visc_upper) * self._slope_unit,
                self.viscosity_to_slope(visc_lower) * self._slope_unit,
            )
            slope_prior = uniform(slope_bounds[0].value, (slope_bounds[1] - slope_bounds[0]).value)
            priors = [D_prior, slope_prior]
        else:
            if len(priors) != 2:
                raise ValueError('Priors must be a list of length 2.')

        # Initialize base class with linear function
        super().__init__(
            data=diffusion,
            function=self._yeh_hummer_function,
            parameter_names=parameter_names,
            parameter_units=parameter_units,
            priors=priors,
            coordinate_name='box_length',
        )

    def _yeh_hummer_function(
        self,
        box_lengths: np.ndarray,
        D_0: float,
        slope: float,
    ) -> np.ndarray:
        """
        Yeh-Hummer function for finite-size correction fit (linear form).
        D_PBC = D_0 - slope / L

        :param box_lengths: Array of box lengths / Angstrom
        :param D_0: Infinite-system diffusion coefficient
        :param slope: i.e., (k_B * T * xi) / (6 * pi * eta)
        """
        return D_0 - slope / np.asarray(box_lengths)

    def _prepare_data_for_fit(self):
        """Prepare data in correct format for fitting."""
        # Convert box lengths to inverse values
        L_values = self.box_lengths.values
        inv_L = 1.0 / L_values

        # Get diffusion values and errors
        D_values = self.diffusion.values
        D_errors = np.sqrt(self.diffusion.variances)

        return inv_L, D_values, D_errors

    def _slope_to_viscosity(self, slope):
        """Convert slope to viscosity using Yeh-Hummer relation."""
        # slope = (k_B * T * xi) / (6 * pi * eta)
        # eta = (k_B * T * xi) / (6 * pi * slope)

        k_B_T = sc.to_unit(const.Boltzmann * self.temperature, 'J')

        # slope has units of [diffusion] * [length]
        slope_with_units = slope * self._slope_unit
        slope_SI = sc.to_unit(slope_with_units, 'm^3/s')

        eta = (k_B_T * self.xi_cubic) / (6 * np.pi * slope_SI)
        return sc.to_unit(eta, 'Pa*s')

    def viscosity_to_slope(self, eta):
        """Convert viscosity to slope for fitting."""
        slope = (const.Boltzmann * self.temperature * self.xi_cubic) / (6 * np.pi * eta)

        # Convert back to data units
        target_unit = self.diffusion.unit * self.box_lengths.unit
        return sc.to_unit(slope, target_unit).value

    def max_likelihood(self):
        """Find maximum likelihood parameters with better initial guess for YehHummer."""
        # Use linear fit for initial parameters
        inv_L, D_values, D_errors = self._prepare_data_for_fit()

        def linear_func(x, a, b):
            return a - b * x

        popt, _ = curve_fit(
            linear_func,
            inv_L,
            D_values,
            sigma=D_errors if np.any(D_errors > 0) else None,
            p0=[np.max(D_values), (D_values[0] - D_values[-1]) / (inv_L[0] - inv_L[-1])],
        )

        D_0_init = popt[0]
        slope_init = popt[1]

        # Use these as initial parameters for optimization
        x0 = [D_0_init, slope_init]

        result = minimize(self.nll, x0, method='L-BFGS-B')

        # Store results
        self.data_group['D_0'] = result.x[0] * self.parameter_units[0]
        self.data_group['slope'] = result.x[1] * self.parameter_units[1]

    def max_aposteriori(self):
        """Find maximum aposteriori parameters with better initial guess for YehHummer."""
        # Use linear fit for initial parameters
        inv_L, D_values, D_errors = self._prepare_data_for_fit()

        def linear_func(x, a, b):
            return a - b * x

        popt, _ = curve_fit(
            linear_func,
            inv_L,
            D_values,
            sigma=D_errors if np.any(D_errors > 0) else None,
            p0=[np.max(D_values), (D_values[0] - D_values[-1]) / (inv_L[0] - inv_L[-1])],
        )

        D_0_init = popt[0]
        slope_init = popt[1]

        # Use these as initial parameters for optimization
        x0 = [D_0_init, slope_init]

        result = minimize(self.nlp, x0, method='L-BFGS-B')

        # Store results
        self.data_group['D_0'] = result.x[0] * self.parameter_units[0]
        self.data_group['slope'] = result.x[1] * self.parameter_units[1]

    @property
    def D_infinite(self) -> sc.Variable | Samples:
        """Return infinite-system diffusion coefficient."""
        return self.data_group['D_0']

    @property
    def shear_viscosity(self) -> sc.Variable | Samples:
        """Return estimated shear viscosity (converted from slope samples)."""
        slope_data = self.data_group['slope']
        if isinstance(slope_data, Samples):
            # eta = (k_B * T * xi) / (6 * pi * slope)
            k_B_T = sc.to_unit(const.Boltzmann * self.temperature, 'J').value
            slope_to_SI = sc.to_unit(1.0 * self._slope_unit, 'm^3/s').value
            slopes_SI = slope_data.values * slope_to_SI
            viscosities = (k_B_T * self.xi_cubic) / (6 * np.pi * slopes_SI)
            return Samples(viscosities, unit=sc.Unit('Pa*s'))
        else:
            return self._slope_to_viscosity(slope_data.value)
