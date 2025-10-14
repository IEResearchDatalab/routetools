import os

from ._version import __version__

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# This disables the preallocation behavior. JAX will instead allocate GPU memory
# as needed, potentially decreasing the overall memory usage. However, this behavior is
# more prone to GPU memory fragmentation, meaning a JAX program that uses most of the
# available GPU memory may OOM with preallocation disabled.
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

__all__ = ["__version__"]
