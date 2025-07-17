"""
Tests for the samples module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import unittest

import numpy as np
import scipp as sc
from scipp.testing import assert_allclose

from kinisi.samples import Samples


class TestSamples(unittest.TestCase):
    """
    Unit tests for the Samples class.
    """

    def test_samples_initialization(self):
        """
        Test the initialization of the Samples class.
        """
        values = np.array([1.0, 2.0, 3.0])
        samples = Samples(values)
        self.assertEqual(samples.dims, ('samples',))
        np.testing.assert_array_equal(samples.values, values)
        self.assertEqual(samples.unit, 'dimensionless')

    def test_samples_unit(self):
        """
        Test the unit of the Samples class.
        """
        values = np.array([1.0, 2.0, 3.0])
        samples = Samples(values, sc.Unit('m'))
        self.assertEqual(samples.unit, sc.units.m)

    def test_samples_repr_html(self):
        """
        Test the HTML representation of the Samples class.
        """
        values = np.array([1.0, 2.0, 3.0])
        samples = Samples(values)
        html_repr = samples._repr_html_()
        self.assertIn('kinisi.Samples', html_repr)

    def test_samples_to_unit(self):
        """
        Test the conversion of samples to a different unit.
        """
        values = np.array([1.0, 2.0, 3.0])
        samples = Samples(values, sc.Unit('m'))
        converted_samples = samples.to_unit(sc.Unit('cm'))
        self.assertEqual(converted_samples.unit, sc.Unit('cm'))
        np.testing.assert_array_equal(converted_samples.values, values * 100)

    def test_parser_datagroup_round_trip(self):
        values = np.array([1.0, 2.0, 3.0])
        samples = Samples(values, sc.Unit('m'))
        dg = samples._to_datagroup()
        samples2 = Samples._from_datagroup(dg)
        assert_allclose(samples, samples2)
