"""
A class to represent samples of a physical quantity using scipp.
This class extends the scipp.Variable class to provide additional functionality
for handling samples, such as calculating the mean and standard deviation of the samples.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import scipp as sc
from bs4 import BeautifulSoup
from uncertainties import ufloat


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

    def _to_datagroup(self):
        """
        Convert the Samples object to a scipp DataGroup for compatibility with other scipp operations.

        :return: A scipp DataGroup containing the Samples object.
        """
        group = {'values': self.values, 'unit': str(self.unit)}
        group['__class__'] = f'{self.__class__.__module__}.{self.__class__.__name__}'
        return sc.DataGroup(group)
    
    @classmethod
    def _from_datagroup(cls, data_group):
        """
        Create a Samples object from a scipp DataGroup.

        :param data_group: A scipp DataGroup containing the samples.

        :return: A Samples object.
        """
        return cls(data_group['values'], unit=data_group['unit'])

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
            preview_div.string = str(ufloat(sc.mean(self).value, sc.std(self, ddof=1).value))

        # Update the type label
        obj_type_divs = soup.find_all('div', class_='sc-obj-type')
        if len(obj_type_divs) > 0:
            parts = obj_type_divs[-1].contents
            if parts:
                parts[0].replace_with('kinisi.Samples')

        return str(soup)

    def to_unit(self, unit: sc.Unit) -> 'Samples':
        """
        Convert the samples to a different unit.

        :param unit: The unit to convert the samples to.

        :return: A new Samples object with the converted values.
        """
        return Samples(sc.to_unit(self, unit).values, unit=unit)
