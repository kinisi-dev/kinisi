"""
The two classes herein enable the determination of the activation energy from a diffusion-Arrhenius or
-Super-Arrhenius plot.
This includes the uncertainty on the activation energy from the MCMC sampling of the plot, with uncertainties on
diffusion.
It is also easy to determine the Bayesian evidence for each of the models with the given data, enabling
the differentiation between data showing Arrhenius and Super-Arrhenius diffusion.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from collections.abc import Callable

import numpy as np
import scipp as sc
from scipp.constants import N_A, R
from scipp.typing import VariableLike

from kinisi.fitting import FittingBase
from kinisi.samples import Samples

R_eV = sc.to_unit(R / N_A, 'eV/K')


class TemperatureDependent(FittingBase):
    """
    A class for temperature-dependent relationships. This class enables MCMC sampling of the
    temperature-dependent relationships and estimation of the Bayesian evidence.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param function: A callable function that describes the relationship between temperature and diffusion.
    :param parameter_names: A tuple of parameter names for the function.
    :param parameter_units: A tuple of sc.Unit objects corresponding to the parameter names.
    :param priors: Optional prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the best fit values.
    """

    def __init__(
        self,
        diffusion,
        function: Callable,
        parameter_names: tuple[str],
        parameter_units: tuple[sc.Unit],
        priors: None | list = None,
    ) -> 'TemperatureDependent':
        self.diffusion = diffusion
        self.temperature = diffusion.coords['temperature']

        super().__init__(
            data=diffusion,
            function=function,
            parameter_names=parameter_names,
            parameter_units=parameter_units,
            priors=priors,
            coordinate_name='temperature',
        )

    def extrapolate(self, extrapolated_temperature: float) -> VariableLike:
        """
        Extrapolate the diffusion coefficient to some un-investigated value. This can also be
        used for interpolation.

        :param extrapolated_temperature: Temperature to return diffusion coefficient at.

        :return: Diffusion coefficient at extrapolated temperature.
        """
        extrapolated_temperature = sc.to_unit(extrapolated_temperature, self.temperature.unit).value
        if isinstance(self.data_group[self.parameter_names[0]], Samples):
            parameters = np.array([self.data_group[name].values for name in self.parameter_names])
            return Samples(self.function(extrapolated_temperature, *parameters), self.diffusion.unit)
        else:
            parameters = np.array([self.data_group[name].value for name in self.parameter_names])
            return sc.scalar(self.function(extrapolated_temperature, *parameters), unit=self.diffusion.unit)


def arrhenius(abscissa: VariableLike, activation_energy: VariableLike, prefactor: VariableLike) -> VariableLike:
    """
    Determine the diffusion coefficient for a given activation energy, and prefactor according to the Arrhenius
    equation.

    :param abscissa: The temperature data.
    :param activation_energy: The activation_energy value.
    :param prefactor: The prefactor value.

    :return: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (R_eV.values * abscissa))


class Arrhenius(TemperatureDependent):
    """
    Evaluate the data with a standard Arrhenius relationship.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param priors: Optional prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the max likelihood values.
    """

    def __init__(
        self,
        diffusion,
        priors: list | None = None,
    ) -> 'Arrhenius':
        parameter_names = ('activation_energy', 'preexponential_factor')
        parameter_units = (sc.Unit('eV'), sc.Unit('cm^2/s'))

        super().__init__(diffusion, arrhenius, parameter_names, parameter_units, priors=priors)

    @property
    def activation_energy(self) -> VariableLike | Samples:
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy']

    @property
    def preexponential_factor(self) -> VariableLike | Samples:
        """
        :return: Preexponential factor.
        """
        return self.data_group['preexponential_factor']


def vtf_equation(
    abscissa: VariableLike, activation_energy: VariableLike, preexponential_factor: VariableLike, T0: VariableLike
) -> VariableLike:
    """
    Evaluate the Vogel-Fulcher-Tammann equation.

    :param abscissa: The temperature data.
    :param activation_energy: The apparent activation energy value.
    :param preexponential_factor: The preexponential factor value.
    :param T0: The T0 value.

    :return: The diffusion coefficient data.
    """
    return preexponential_factor * np.exp(-activation_energy / (R_eV.values * (abscissa - T0)))


class VogelFulcherTammann(TemperatureDependent):
    """
    Evaluate the data with a Vogel-Fulcher-Tammann relationship.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param priors: Optional prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the max likelihood values.
    """

    def __init__(
        self,
        diffusion,
        priors: list | None = None,
    ) -> 'VogelFulcherTammann':
        parameter_names = ('activation_energy', 'preexponential_factor', 'T0')
        parameter_units = (sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'))

        super().__init__(diffusion, vtf_equation, parameter_names, parameter_units, priors=priors)

    @property
    def activation_energy(self) -> VariableLike | Samples:
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy']

    @property
    def preexponential_factor(self) -> VariableLike | Samples:
        """
        :return: Preexponential factor.
        """
        return self.data_group['preexponential_factor']

    @property
    def T0(self) -> VariableLike | Samples:
        """
        :return: Temperature factor for the VTF equation in kelvin.
        """
        return self.data_group['T0']


def piecewise_smooth_equation(
    abscissa: VariableLike,
    activation_energy_low: VariableLike,
    activation_energy_high: VariableLike,
    preexponential_factor: VariableLike,
    T0: VariableLike,
    width: VariableLike,
) -> VariableLike:
    """
    Evaluate the data with two Arrhenius regions following a piecewise equation smoothed with a sigmoid function.

    :param abscissa: The temperature data.
    :param activation_energy_high: The high-temperature activation energy value.
    :param activation_energy_low: The low-temperature activation energy value.
    :param prefactor: The prefactor value.
    :param T0: Phase transition temperature.
    :param width: width of the sigmoid transition.

    :return: The diffusion coefficient data.
    """
    h = 1.0 / (1.0 + np.exp((abscissa - T0) / width))
    activation_energy_delta = activation_energy_low - activation_energy_high
    return preexponential_factor * np.exp(
        -(activation_energy_high + h * activation_energy_delta) / (R_eV.values) * (1 / abscissa - 1 / T0)
    )


class PiecewiseSmoothArrhenius(TemperatureDependent):
    """
    Evaluate the data with two Arrhenius regions following a piecewise relationship smoothed with a sigmoid function.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param priors: Optional prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the best fit values.
    """

    def __init__(
        self,
        diffusion,
        priors: list | None = None,
    ) -> 'PiecewiseSmoothArrhenius':
        parameter_names = ('activation_energy_low', 'activation_energy_high', 'preexponential_factor', 'T0', 'width')
        parameter_units = (sc.Unit('eV'), sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'), sc.Unit('K'))

        super().__init__(diffusion, piecewise_smooth_equation, parameter_names, parameter_units, priors=priors)

    @property
    def activation_energy_low_temperature(self) -> VariableLike | Samples:
        """
        :return: Low-temperature activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy_low']

    @property
    def activation_energy_high_temperature(self) -> VariableLike | Samples:
        """
        :return: High-temperature activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy_high']

    @property
    def preexponential_factor(self) -> VariableLike | Samples:
        """
        :return: Preexponential factor.
        """
        return self.data_group['preexponential_factor']

    @property
    def T0(self) -> VariableLike | Samples:
        """
        :return: Transition temperature in between the low temperature and high temperature phases in kelvin.
        """
        return self.data_group['T0']

    @property
    def width(self) -> VariableLike | Samples:
        """
        :return: Width of the sigmoid function
        """
        return self.data_group['width']


def piecewise_equation(
    abscissa: VariableLike,
    activation_energy_low: VariableLike,
    activation_energy_high: VariableLike,
    preexponential_factor: VariableLike,
    T0: VariableLike,
) -> VariableLike:
    """
    Evaluate the data with two Arrhenius regions following a piecewise equation.

    :param abscissa: The temperature data.
    :param activation_energy_high: The high-temperature activation energy value.
    :param activation_energy_low: The low-temperature activation energy value.
    :param prefactor: The prefactor value.
    :param T0: Phase transition temperature.

    :return: The diffusion coefficient data.
    """
    h = np.where(abscissa < T0, 1.0, 0.0)
    activation_energy_delta = activation_energy_low - activation_energy_high
    return preexponential_factor * np.exp(
        -(activation_energy_high + h * activation_energy_delta) / (R_eV.values) * (1 / abscissa - 1 / T0)
    )


class PiecewiseArrhenius(TemperatureDependent):
    """
    Evaluate the data with two Arrhenius regions following a piecewise relationship.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param priors: Optional prior probability distributions for the parameters of the function.
        Defaults to None, in which case a uniform distribution is defined with limits of
        +/- 50 percent of the best fit values.
    """

    def __init__(
        self,
        diffusion,
        priors: list | None = None,
    ) -> 'PiecewiseArrhenius':
        parameter_names = ('activation_energy_low', 'activation_energy_high', 'preexponential_factor', 'T0')
        parameter_units = (sc.Unit('eV'), sc.Unit('eV'), sc.Unit('cm^2/s'), sc.Unit('K'))

        super().__init__(diffusion, piecewise_equation, parameter_names, parameter_units, priors=priors)

    @property
    def activation_energy_low_temperature(self) -> VariableLike | Samples:
        """
        :return: Low-temperature activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy_low']

    @property
    def activation_energy_high_temperature(self) -> VariableLike | Samples:
        """
        :return: High-temperature activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy_high']

    @property
    def preexponential_factor(self) -> VariableLike | Samples:
        """
        :return: Preexponential factor.
        """
        return self.data_group['preexponential_factor']

    @property
    def T0(self) -> VariableLike | Samples:
        """
        :return: Transition temperature in between the low temperature and high temperature phases in kelvin.
        """
        return self.data_group['T0']
