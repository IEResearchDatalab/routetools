FROM python:3.11-slim

# Pre-install scoring dependencies from pinned requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
