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
from scipy.optimize import curve_fit

from kinisi import __version__
from kinisi.fitting import FittingBase

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

    :param diffusion: sc.DataArray with diffusion coefficients and box_length coordinate
    :param temperature: Temperature (will be extracted from coords if not provided)
    :param bounds: Optional bounds for [D_0, viscosity] parameters
    """

    def __init__(self, diffusion, temperature: sc.Variable, bounds=None):
        self.diffusion = diffusion

        # Extract box lengths from coordinates
        self.box_lengths = diffusion.coords['box_length']
        self.temperature = temperature

        # Constants
        self.xi_cubic = 2.837297  # Ewald constant for cubic boxes
        # Set up parameters for YehHummer fitting
        parameter_names = ('D_0', 'viscosity')
        parameter_units = (diffusion.unit, sc.Unit('Pa*s'))

        # Set default bounds if not provided
        if bounds is None:
            # Auto-generate reasonable bounds
            D_max = np.max(diffusion.values)
            bounds = (
                (D_max * 0.8 * diffusion.unit, D_max * 2.0 * diffusion.unit),
                (1e-5 * sc.Unit('Pa*s'), 1e-1 * sc.Unit('Pa*s')),
            )

        # Initialize base class with custom function
        super().__init__(
            data=diffusion,
            function=self._yeh_hummer_function,
            parameter_names=parameter_names,
            parameter_units=parameter_units,
            bounds=bounds,
            coordinate_name='box_length',
        )

        # Convert bounds to values for optimization (for backward compatibility)
        self.bounds_values = tuple(
            [
                (b[0].to(unit=u).value, b[1].to(unit=u).value)
                for b, u in zip(self.bounds, self.parameter_units, strict=False)
            ]
        )

    def _yeh_hummer_function(
        self,
        box_lengths: np.ndarray,
        D_0: float,
        viscosity: float,
    ) -> np.ndarray:
        """
        Yeh-Hummer function for finitie-size correction fit.

        :param box_lengths: Array of box lengths / Å
        :param D_0: Infinite-system diffusion coefficient
        :param viscosity: Shear viscosity
        """
        # Handle both scalar and array inputs
        box_lengths = np.asarray(box_lengths)
        viscosity = np.asarray(viscosity)

        inv_L = 1.0 / box_lengths

        if viscosity.ndim == 0:
            eta_with_unit = viscosity * self.parameter_units[1]
            slope = self.viscosity_to_slope(eta_with_unit)
        else:
            # viscosity as an array (from MCMC samples)
            slopes = []
            for visc_val in viscosity:
                eta_with_unit = visc_val * self.parameter_units[1]
                slope = self.viscosity_to_slope(eta_with_unit)
                slopes.append(slope)
            slope = np.array(slopes)

        return self.yeh_hummer_linear(inv_L, D_0, slope)

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

        # slope has units of [diffusion] / [1/length] = [diffusion] * [length]
        # diffusion is cm^2/s, box_lengths is Å, so slope * diffusion.unit / (1/box_lengths.unit)
        # This gives us (cm^2/s) / (1/Å) = cm^2/s * Å = cm^2 * Å / s
        slope_with_units = slope * self.diffusion.unit / (1 / self.box_lengths.unit)
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

        # Convert slope to viscosity
        eta_init = self._slope_to_viscosity(slope_init).value

        # Use these as initial parameters for optimization
        x0 = [D_0_init, eta_init]

        from scipy.optimize import minimize

        # Convert bounds to format expected by scipy
        bounds_scipy = [(b[0].value, b[1].value) for b in self.bounds]
        result = minimize(self.nll, x0, bounds=bounds_scipy, method='L-BFGS-B')

        # Store results
        self.data_group['D_0'] = result.x[0] * self.parameter_units[0]
        self.data_group['viscosity'] = result.x[1] * self.parameter_units[1]

    @property
    def D_infinite(self):
        """Return infinite-system diffusion coefficient."""
        return self.data_group['D_0']

    @property
    def shear_viscosity(self):
        """Return estimated shear viscosity."""
        return self.data_group['viscosity']

    @staticmethod
    def yeh_hummer_linear(inv_L, D_0, slope):
        """
        Linear form of Yeh-Hummer equation for fitting.

        D_PBC = D_0 - slope * (1/L)

        where slope = (k_B * T * xi) / (6 * pi * eta)

        :param inv_L: Inverse box lengths (1/L)
        :param D_0: Infinite-system diffusion coefficient
        :param slope: Slope containing viscosity information
        :return: D_PBC values
        """
        return D_0 - slope * inv_L
