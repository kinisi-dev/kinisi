__version__ = '2.0.2'

from .due import Doi, due
from .species import Species

due.cite(
    Doi('https://doi.org/10.21105/joss.05984'),
    description='kinisi: Bayesian analysis of mass transport from molecular dynamics simulations',
    path='kinisi',
)
