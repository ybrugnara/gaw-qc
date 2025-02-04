# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /application


RUN mkdir /data


# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.

COPY pyproject.toml /application/

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=/pyproject.toml,target=/application/pyproject.toml \
    pip install -e .


COPY src/ /application/


RUN mkdir /assets
COPY src/gaw_qc/assets/css /assets/css
COPY src/gaw_qc/assets/logos /assets/logos
COPY src/gaw_qc/assets/favicon.ico /assets/
RUN mkdir /assets/images
COPY src/gaw_qc/assets/images/map_gawsis.png /assets/images/
RUN rm -r gaw_qc/assets


# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "-p", "80", "--timeout", "60", "gaw_qc.wsgi:app"]
