"""
A class to represent samples of a physical quantity using scipp.
This class extends the scipp.Variable class to provide additional functionality
for handling samples, such as calculating the mean and standard deviation of the samples.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import scipp as sc
from uncertainties import ufloat
from bs4 import BeautifulSoup


class Samples(sc.Variable):
    """
    A subclass of scipp.Variable that represents samples of a physical quantity.
    This class is designed to add some specific functionality for handling
    samples, such as calculating the mean and standard deviation of the samples.
    It also overrides the HTML representation to include these statistics.

    :param values: The values of the samples.
    :param unit: The unit of the samples, if applicable. Optional, defaults to dimensionless.
    """

    def __init__(self, values, unit=sc.units.dimensionless):
        super().__init__(values=values, unit=unit, dims=['samples'])

    def _repr_html_(self) -> str:
        """
        This function augments the default HTML representation of a scipp Variable
        to include the mean and standard deviation of the samples.

        :return: A string containing the HTML representation of the Samples object,
                 including the mean and standard deviation.
        """
        html = sc.make_html(self)
        soup = BeautifulSoup(html, 'html.parser')

        # Update the preview value
        preview_div = soup.find('div', class_='sc-value-preview sc-preview')
        if preview_div:
            preview_div.string = str(
                ufloat(sc.mean(self).value, sc.std(self, ddof=1).value)
            )

        # Update the type label
        obj_type_divs = soup.find_all('div', class_='sc-obj-type')
        if len(obj_type_divs) > 0:
            parts = obj_type_divs[-1].contents
            if parts:
                parts[0].replace_with("kinisi.Samples")

        return str(soup)

    def to_unit(self, unit: sc.Unit) -> 'Samples':
        """
        Convert the samples to a different unit.

        :param unit: The unit to convert the samples to.

        :return: A new Samples object with the converted values.
        """
        return Samples(sc.to_unit(self, unit).values, unit=unit)
