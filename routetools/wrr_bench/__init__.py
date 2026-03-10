"""Legacy ocean benchmark module.

.. deprecated::
    ``wrr_bench`` is superseded by :mod:`routetools.era5` which provides
    real-world ERA5 wind, wave, and current fields.  ``wrr_bench`` will be
    removed in a future release.  Migrate to the ERA5-based pipeline::

        from routetools.era5 import load_era5_windfield, load_era5_wavefield
"""

import warnings

warnings.warn(
    "routetools.wrr_bench is deprecated and will be removed in a future release. "
    "Use routetools.era5 instead for real-world weather data.",
    DeprecationWarning,
    stacklevel=2,
)

from routetools.wrr_bench.load import load_real_instance
from routetools.wrr_bench.ocean import Ocean

__all__ = ["load_real_instance", "Ocean"]
