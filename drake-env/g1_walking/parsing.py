"""Helpers for loading the model files shipped with g1_walking."""

from pathlib import Path

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

PACKAGE_NAME = "g1_walking"
PACKAGE_ROOT = Path(__file__).parent


def make_parser(plant: MultibodyPlant) -> Parser:
    """Create a parser for the plant that can resolve g1_walking models."""
    parser = Parser(plant)
    package_map = parser.package_map()
    if not package_map.Contains(PACKAGE_NAME):
        package_map.Add(PACKAGE_NAME, str(PACKAGE_ROOT))
    return parser
