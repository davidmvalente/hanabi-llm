"""Hanabi for tool-calling language models, driven by an RDF world model."""

# Importing these modules registers every ontology-named function in
# ``registry.FUNCTIONS``; ``Game`` verifies the set is complete.
from . import engine as engine
from . import tools as tools  # noqa: F401  (registers the read tools)

__all__ = ["__version__"]
__version__ = "0.1.0"
