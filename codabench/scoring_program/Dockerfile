FROM python:3.11-slim

# Pre-install scoring dependencies so submissions don't wait for pip install
RUN pip install --no-cache-dir numpy netCDF4 pyshp shapely matplotlib
