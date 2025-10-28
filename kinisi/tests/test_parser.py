"""
Tests for parser module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61), Josh Dunn (jd15489) & Oskar G. Soulas (osoulas)
# pylint: disable=R0201

import unittest

import MDAnalysis as mda
import numpy as np
import pytest
import scipp as sc
from numpy.testing import assert_almost_equal, assert_equal

from kinisi import parser
from kinisi.tests import TEST_FILE_PATH


class mda_universe_generator:
    def __init__(self, coords, weights):
        self.coords = coords.values
        self.weights = np.tile(weights, 2)
        self.bonds = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
        ]
        self.box_dimensions = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]
        self.types = ['1'] * 10

        u = mda.Universe.empty(10, 2, atom_resindex=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], trajectory=True)
        u.add_TopologyAttr('masses')
        u.load_new(self.coords, order='fac')
        u.add_TopologyAttr('types')
        u.add_TopologyAttr('bonds', self.bonds)
        u.atoms.masses = self.weights
        u.atoms.types = self.types
        self.u = u

    def calculate_com(self):
        mda_com_coords = np.zeros((10, 2, 3))
        for i, _ts in enumerate(self.u.trajectory):
            self.u.dimensions = self.box_dimensions
            _groups = self.u.select_atoms('type 1').fragments
            positions = np.array([g.center_of_mass(unwrap=True) for g in _groups], dtype=np.float32)
            mda_com_coords[i] = positions
        return mda_com_coords


class Test_calculate_centers_of_mass(unittest.TestCase):
    seeds = [42, 1998, 7, 64, 11]
    for x in seeds:
        np.random.seed(x)
        coords_l = np.random.rand(10, 10, 3) * 0.5
        coords = sc.array(dims=['time', 'particle', 'dimension'], values=coords_l, unit=sc.units.dimensionless)
        indices = sc.array(dims=['particles', 'atoms in particle'], values=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        weights = sc.array(dims=['atoms in particle'], values=np.random.rand(5) * 6)

        kinisi_com = parser._calculate_centers_of_mass(coords=coords, weights=weights, indices=indices).values

        mda_object = mda_universe_generator(coords=coords, weights=weights)
        mda_com_coords = mda_object.calculate_com()

        assert_almost_equal(mda_com_coords, kinisi_com)


class TestSubsetApprox(unittest.TestCase):
    """
    Unit tests for the subset approximation functionality.
    """

    def test_is_subset_approx(self):
        data = np.array([1, 2, 3, 4, 5])
        subset = np.array([1, 3, 5])
        assert parser.is_subset_approx(subset, data)

    def test_is_subset_approx_fail(self):
        data = np.array([1, 2, 3, 4, 5])
        subset = np.array([1, 3, 5, 7])
        assert not parser.is_subset_approx(subset, data)


class test_is_orthorhombic(unittest.TestCase):
    """
    Unit tests for checking cell shapes.
    """

    def test_is_orthorhombic(self):
        latt = np.tile([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 1, 1))
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        assert parser.is_orthorhombic(latt)

    def test_is_orthorhombic_close(self):
        latt = np.tile(
            [[1, 0, 0], [1 * np.cos(90 * (np.pi / 180)), 1 * np.sin(90 * np.pi / 180), 0], [0, 0, 1]], (3, 1, 1)
        )
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        assert parser.is_orthorhombic(latt)

    def test_is_not_orthorhombic(self):
        latt = np.tile(
            [[1, 0, 0], [1 * np.cos(60 * (np.pi / 180)), 1 * np.sin(60 * np.pi / 180), 0], [0, 0, 1]], (3, 1, 1)
        )
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        assert not parser.is_orthorhombic(latt)

    def test_some_is_not_orthorhombic(self):
        latt = np.tile([[1, 0, 0], [1, 1, 0], [0, 0, 1]], (3, 1, 1))
        latt = np.concatenate(
            (latt, [[[1, 0, 0], [1 * np.cos(60 * (np.pi / 180)), 1 * np.sin(60 * np.pi / 180), 0], [0, 0, 1]]]), axis=0
        )
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        assert not parser.is_orthorhombic(latt)

    def test_orthorhombic_calculate_displacements(self):
        coords = [
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.9, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.9, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.1, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.9, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.9, 0.9]],
        ]
        coords = sc.array(dims=['time', 'particle', 'dimension'], values=coords, unit=sc.units.dimensionless)
        latt = np.tile([[10, 0, 0], [0, 10, 0], [0, 0, 10]], (coords.shape[0], 1, 1))
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        disp = parser.Parser.orthorhombic_calculate_displacements(coords=coords, lattice=latt)
        test_disp = [
            [[0.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[0.0, -2.0, 0.0]],
            [[0.0, 2.0, 0.0]],
            [[0.0, 0.0, -2.0]],
            [[0.0, 0.0, 2.0]],
            [[-2.0, -2.0, 0.0]],
            [[2.0, 2.0, 0.0]],
            [[-2.0, 0.0, -2.0]],
            [[2.0, 0.0, 2.0]],
            [[0.0, -2.0, -2.0]],
            [[0.0, 2.0, 2.0]],
            [[-2.0, -2.0, -2.0]],
        ]
        test_disp = sc.array(
            dims=['obs', 'particle', 'dimension'], values=np.cumsum(test_disp, axis=0), unit=sc.units.angstrom
        )
        assert_almost_equal(disp.values, test_disp.values)

    def test_non_orthorhombic_calculate_displacements(self):
        coords = [
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.9, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.9, 0.1]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.1, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.9, 0.9]],
            [[0.1, 0.1, 0.1]],
            [[0.9, 0.9, 0.9]],
        ]
        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords, unit=sc.units.dimensionless)
        latt = np.tile([[10, 0, 0], [0, 10, 0], [0, 0, 10]], (coords.shape[0], 1, 1))
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=sc.units.angstrom)
        disp = parser.Parser.non_orthorhombic_calculate_displacements(coords=coords, lattice=latt)
        test_disp = [
            [[0.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[0.0, -2.0, 0.0]],
            [[0.0, 2.0, 0.0]],
            [[0.0, 0.0, -2.0]],
            [[0.0, 0.0, 2.0]],
            [[-2.0, -2.0, 0.0]],
            [[2.0, 2.0, 0.0]],
            [[-2.0, 0.0, -2.0]],
            [[2.0, 0.0, 2.0]],
            [[0.0, -2.0, -2.0]],
            [[0.0, 2.0, 2.0]],
            [[-2.0, -2.0, -2.0]],
        ]
        test_disp = sc.array(
            dims=['obs', 'atom', 'dimension'], values=np.cumsum(test_disp, axis=0), unit=sc.units.angstrom
        )
        assert_almost_equal(disp.values, test_disp.values)


dg = sc.io.load_hdf5(TEST_FILE_PATH / 'example_drift.h5')
coords = dg['coords']
latt = dg['latt']
time_step = dg['time_step']
step_skip = dg['step_skip']
dt = dg['dt']
specie_indices = dg['specie_indices']
drift_indices = dg['drift_indices']
dimension = dg['dimension']
disp = dg['disp']


class TestParser(unittest.TestCase):
    """
    Unit tests for the Parser class
    """

    def test_parser_init_time_interval(self):
        data = parser.Parser(coords, latt, time_step, step_skip, dt, specie_indices, drift_indices, dimension=dimension)
        assert_equal(data.time_step, time_step)

    def test_parser_init_stepskip(self):
        data = parser.Parser(coords, latt, time_step, step_skip, dt, specie_indices, drift_indices, dimension=dimension)
        assert_equal(data.step_skip, step_skip)

    def test_parser_init_drift_indices(self):
        data = parser.Parser(coords, latt, time_step, step_skip, dt, specie_indices, drift_indices, dimension=dimension)
        assert_equal(data.drift_indices.values, drift_indices.values)

    def test_parser_dt(self):
        data = parser.Parser(coords, latt, time_step, step_skip, dt, specie_indices, drift_indices, dimension=dimension)
        assert_equal(data.dt.size, 140)

    def test_parser_datagroup_round_trip(self):
        data = parser.Parser(coords, latt, time_step, step_skip, dt, specie_indices, drift_indices, dimension=dimension)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert vars(data) == vars(data_2)
        assert type(data) is type(data_2)

    errors_for_get_specie_and_drift_indices = [
        (None, None, None),
        ('Li', sc.array(dims=['atom'], values=[1, 2, 3]), None),
        ('Li', sc.array(dims=['atom'], values=[1, 2, 3]), [4, 5, 6]),
        (sc.array(dims=['specie'], values=['Li', 'Na']), sc.array(dims=['atom'], values=[1, 2, 3]), None),
        (
            sc.array(dims=['specie'], values=['Li', 'Na']),
            sc.array(dims=['atom'], values=[1, 2, 3]),
            sc.array(dims=['atom'], values=[1, 2, 3]),
        ),
        (None, None, sc.array(dims=['atom'], values=[1, 2, 3])),
        (sc.array(dims=['specie'], values=['Li', 'Na']), None, None),
    ]

    @pytest.mark.parametrize('test_input', errors_for_get_specie_and_drift_indices)
    def test_get_specie_and_drift_indices_errors(test_input):
        with pytest.raises(TypeError):
            parser.Parser.get_specie_and_drift_indices(None, *test_input, [])
