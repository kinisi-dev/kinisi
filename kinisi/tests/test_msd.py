# """
# Longer tests for application inluding MSD 
# """

# # Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# # Distributed under the terms of the MIT License
# # author: Andrew R. McCluskey (arm61)
# # pylint: disable=R0201

# import unittest
# import numpy as np
# from numpy.testing import assert_almost_equal
# import os
# import kinisi

# from kinisi.analyze import DiffusionAnalyzer
# from ase.io import read


# class TestMSD(unittest.TestCase):
#     """
#     Longer tests for application
#     """

#     def test_msd_from_ase(self):
#         atoms = read(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/LiPS.exyz'),
#                      format='extxyz',
#                      index=':')
#         params = {
#             'specie': 'Li',
#             'time_step': 0.001,
#             'step_skip': 20,
#             'n_steps': 250,
#             'sampling': 'multi-origin',
#             'progress': False
#         }
#         xyz = {'dimension': 'xyz', 'bootstrap': False, 'progress': False}
#         msd_from_ase = DiffusionAnalyzer.from_ase(atoms, parser_params=params, uncertainty_params=xyz).msd
#         assert_almost_equal(msd_from_ase,
#                             np.load(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/LiPS_msd.npy')))
