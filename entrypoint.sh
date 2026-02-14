#!/bin/bash
# Entrypoint: runs as root to handle bind-mount directories,
# then drops to appuser for the application process.

# Ensure data directories exist and are owned by appuser
# (handles bind-mount scenarios where Docker creates the host
#  directory as root before the container starts)
mkdir -p /code/data/uploads
chown -R appuser:appuser /code/data

# Drop to non-root user and exec the CMD
exec gosu appuser "$@"
