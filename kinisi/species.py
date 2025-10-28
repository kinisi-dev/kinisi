import dataclasses


@dataclasses.dataclass
class Species:
    """
    Represents a species with associated indices, masses, and charge.

    Attributes
    ----------
    indices : list[list[int]]
        A list of molecules, where each molecule is represented by a list of atom indices
        in the trajectory.
    masses : list[float]
        A list of masses corresponding to each group of indices.
        For COM calculations, the order of the atoms inside each molecule must be consistent.
    charge : float, optional
        The charge per molecule. Defaults to 0.0.
    """
    indices: list[list[int]]
    masses: list[float]
    charge: float = 0.0

    def __post_init__(self):
        if not all(len(group) == len(self.masses) for group in self.indices):
            raise ValueError("Each group of indices must correspond to a mass in masses.")
