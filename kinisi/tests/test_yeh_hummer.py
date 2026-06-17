"""
Tests for the YehHummer finite-size correction module.
"""

import numpy as np
import pytest
import scipp as sc
from scipy.stats import uniform

from kinisi.yeh_hummer import YehHummer


class TestYehHummer:
    """Tests for the YehHummer class."""

    def test_yeh_hummer_function(self):
        """Test the Yeh-Hummer fitting function."""
        box_lengths = np.array([20.0, 30.0, 40.0])
        D_values = np.array([5.0e-5, 5.2e-5, 5.4e-5])
        D_errors = np.array([0.1e-5, 0.1e-5, 0.1e-5])

        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        yh = YehHummer(td, temperature=sc.scalar(298, unit='K'))

        # Test the internal function: D_PBC = D_0 - slope / L
        D_0 = 6.0e-5
        slope = 1.0e-6
        result = yh._yeh_hummer_function(box_lengths, D_0, slope)
        expected = D_0 - slope / box_lengths

        np.testing.assert_array_almost_equal(result, expected)

    def test_yeh_hummer_tip3p_data(self):
        """Test YehHummer with TIP3P water data from the example."""
        # TIP3P water data from Yeh & Hummer paper (corrected values)
        box_lengths = np.array([18.58, 23.42, 29.51, 37.19, 46.86])  # Angstroms
        D_values = np.array([4.884e-5, 5.123e-5, 5.315e-5, 5.466e-5, 5.590e-5])  # cm^2/s
        D_errors = np.array([0.032e-5, 0.027e-5, 0.014e-5, 0.011e-5, 0.013e-5])  # cm^2/s

        # Create DataArray
        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        # Create YehHummer object
        yh = YehHummer(td, temperature=sc.scalar(298, unit='K'))

        # Check that parameters are reasonable
        assert yh.D_infinite.value > 0
        assert yh.shear_viscosity.value > 0

        # Check units
        assert yh.D_infinite.unit == sc.Unit('cm^2/s')
        assert yh.shear_viscosity.unit == sc.Unit('Pa*s')

        # Check that D_infinite is larger than any measured value (should be extrapolated to infinite size)
        assert yh.D_infinite.value > np.max(D_values)

        # Check that viscosity is in reasonable range for water at 298K (should be around 1e-3 Pa*s)
        print(yh.shear_viscosity)
        print('a')        
        print(yh.shear_viscosity.value)
        assert 1e-4 < yh.shear_viscosity.value < 1e-2

    def test_yeh_hummer_mcmc(self):
        """Test MCMC functionality with reduced sample size."""
        # Use smaller dataset for faster testing
        box_lengths = np.array([20.0, 30.0, 40.0])  # Angstroms
        D_values = np.array([5.0e-5, 5.2e-5, 5.4e-5])  # cm^2/s
        D_errors = np.array([0.1e-5, 0.1e-5, 0.1e-5])  # cm^2/s

        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        yh = YehHummer(td, temperature=sc.scalar(298, unit='K'))

        # Run MCMC with small sample size for testing
        yh.mcmc(n_samples=50, n_walkers=8, n_burn=20, n_thin=2)

        # Check that we get Samples objects
        from kinisi.samples import Samples

        assert isinstance(yh.D_infinite, Samples)
        assert isinstance(yh.shear_viscosity, Samples)

        # Check that distribution property works
        dist = yh.distribution
        assert dist.shape[0] == len(box_lengths)  # Number of data points
        assert dist.shape[1] > 0  # Number of samples

    def test_yeh_hummer_priors(self):
        """Test YehHummer with custom priors."""
        box_lengths = np.array([20.0, 30.0, 40.0])
        D_values = np.array([5.0e-5, 5.2e-5, 5.4e-5])
        D_errors = np.array([0.1e-5, 0.1e-5, 0.1e-5])

        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        # Custom priors
        priors = (
            uniform(4e-5, 7e-5),  # D_0 prior
            uniform(1e-4, 1e-2),  # viscosity prior
        )

        yh = YehHummer(td, temperature=sc.scalar(298, unit='K'), priors=priors)

        # Check that fitted values are within priors
        assert priors[0].a <= yh.D_infinite.value <= (priors[0].b + priors[0].a)
        assert priors[1].a <= yh.shear_viscosity.value <= (priors[1].b + priors[1].a)

    def test_yeh_hummer_properties(self):
        """Test YehHummer property accessors."""
        box_lengths = np.array([20.0, 30.0, 40.0])
        D_values = np.array([5.0e-5, 5.2e-5, 5.4e-5])
        D_errors = np.array([0.1e-5, 0.1e-5, 0.1e-5])

        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        yh = YehHummer(td, temperature=sc.scalar(298, unit='K'))

        # Test property accessors
        assert yh.D_infinite == yh.data_group['D_0']
        # slope is stored in data_group, viscosity is computed from slope
        assert 'slope' in yh.data_group.keys()
        assert yh.shear_viscosity.unit == sc.Unit('Pa*s')

        # Test that the object has string representations
        assert len(str(yh)) > 0
        assert len(repr(yh)) > 0

    def test_yeh_hummer_invalid_priors(self):
        """Test YehHummer with invalid priors."""
        box_lengths = np.array([20.0, 30.0, 40.0])
        D_values = np.array([5.0e-5, 5.2e-5, 5.4e-5])
        D_errors = np.array([0.1e-5, 0.1e-5, 0.1e-5])

        td = sc.DataArray(
            data=sc.array(dims=['system'], values=D_values, variances=D_errors**2, unit='cm^2/s'),
            coords={'box_length': sc.Variable(dims=['system'], values=box_lengths, unit='angstrom')},
        )

        # Wrong number of bounds
        priors = [uniform(4e-5, 7e-5)]  # Only one bound

        with pytest.raises(ValueError, match='Priors must be a list of length 2.'):
            YehHummer(td, temperature=sc.scalar(298, unit='K'), priors=priors)
