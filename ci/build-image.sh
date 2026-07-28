#!/usr/bin/env bash
# Build the pdmp CI image on the GitLab runner host.
#
# gitlab.tudelft.nl has no container registry, so the image lives only in this
# host's Docker daemon and .gitlab-ci.yml pulls it with pull_policy=if-not-present.
#
# Usage, on the runner host:
#   cd /srv/pdmp-ci && git pull && sudo ci/build-image.sh
#
# Or in one shot from a workstation:
#   ssh <host> 'cd /srv/pdmp-ci && git pull && sudo ci/build-image.sh'
#
# Keep the clone owned by your user: `git pull` must NOT run under sudo (root has
# no GitLab SSH key, and git refuses pulls into a repo it sees as another user's).
# sudo here is only for access to the Docker socket.
#
# Afterwards, bump the `image:` tag in .gitlab-ci.yml to the tag printed below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
TAG="${1:-$(date +%Y-%m)}"

echo "=== Building pdmp CI image ==="
echo "    context:  ${REPO_DIR}"
echo "    tags:     pdmp-ci:${TAG}, pdmp-ci:latest"
echo ""

docker build -t "pdmp-ci:${TAG}" -t pdmp-ci:latest "${REPO_DIR}"

# /data is only 49 GB and shared with the GitLab install; don't leave the build
# cache or the layers orphaned by this rebuild lying around.
docker builder prune -af
docker image prune -f

echo ""
echo "=== Build complete: pdmp-ci:${TAG} ==="
echo "Set 'image: pdmp-ci:${TAG}' in .gitlab-ci.yml, then commit and push."
