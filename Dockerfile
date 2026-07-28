# CI image for pdmp: conda environment + jax-fem, baked in.
#
# gitlab.tudelft.nl has no container registry, so this image lives only in the
# runner host's Docker daemon, and .gitlab-ci.yml consumes it with the runner's
# pull_policy = "if-not-present".
#
# Rebuild whenever environment.yml or the pinned jax-fem commit changes, via the
# repo clone kept on the runner host at /srv/pdmp-ci:
#
#   ssh <host> 'cd /srv/pdmp-ci && git pull && sudo ci/build-image.sh'
#
# The clone is user-owned (sudo is only needed for the Docker socket); cloning it
# as root would use root's SSH keys, which GitLab does not know.
#
# Then bump the `image:` tag in .gitlab-ci.yml to the tag the script prints.

FROM condaforge/mambaforge:24.9.0-0

ENV DEBIAN_FRONTEND=noninteractive

# System libraries required by gmsh's bundled binary.
# Determined via: ldd libgmsh.so | grep -v conda
# Covers: OpenGL, X11/Xft/Xinerama, freetype, png, brotli, gomp.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git \
        libgomp1 \
        libgl1-mesa-glx \
        libglu1-mesa \
        libopengl0 \
        libx11-6 \
        libxext6 \
        libxft2 \
        libxrender1 \
        libxinerama1 \
        libxcursor1 \
        libxrandr2 \
        libxi6 \
        libfontconfig1 \
        libfreetype6 \
        libpng16-16 \
        libbrotli1 \
    && rm -rf /var/lib/apt/lists/*

# Environment name (pdmp-jax) and all dependencies come from environment.yml.
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
    && mamba clean --all -y \
    && rm /tmp/environment.yml

ENV PDMP_ENV=/opt/conda/envs/pdmp-jax

# jax-fem at the commit pinned by the project (see CLAUDE.md).
RUN git clone https://github.com/deepmodeling/jax-fem.git /tmp/jax-fem \
    && cd /tmp/jax-fem \
    && git checkout c3fbcb3 \
    && "$PDMP_ENV/bin/pip" install --no-cache-dir /tmp/jax-fem \
    && "$PDMP_ENV/bin/pip" cache purge || true \
    && rm -rf /tmp/jax-fem

# Put the environment on PATH so CI scripts need no `mamba run` wrapper.
ENV PATH=/opt/conda/envs/pdmp-jax/bin:$PATH \
    CONDA_DEFAULT_ENV=pdmp-jax \
    CONDA_PREFIX=/opt/conda/envs/pdmp-jax \
    DISPLAY=""

WORKDIR /workspace
