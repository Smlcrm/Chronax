"""
Theta models subpackage.

Re-exports the public Theta forecasters so callers can do:

    from chronax.models.theta import Theta, AutoTheta
"""

# from .theta_model import Theta
from .auto_theta import AutoTheta, Theta

__all__ = ["Theta", "AutoTheta"]

