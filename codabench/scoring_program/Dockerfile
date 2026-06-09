FROM python:3.11-slim

# Pre-install scoring dependencies from pinned requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install RISE performance model wheel (optional comparison in scorer)
COPY swopp3_performance_model-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/swopp3_performance_model-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && rm /tmp/*.whl
