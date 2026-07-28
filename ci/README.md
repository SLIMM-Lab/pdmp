# CI Setup

## Overview

Tests run inside `pdmp-ci`, a Docker image with the full conda environment and
`jax-fem` baked in. CI does nothing but `pytest --cov=pdmp`.

Previously the pipeline ran `mamba env create` on every push. That solved and
extracted roughly 8 GB into the job container on each run, which repeatedly
filled the runner host's disk and broke the pipeline. Baking the environment
into an image removed that cost: jobs went from ~15 minutes to under a minute.

**`gitlab.tudelft.nl` has no container registry.** The image therefore exists
only in the runner host's local Docker daemon — it is never pushed or pulled.
The runner is configured with `pull_policy = "if-not-present"` so it uses that
local copy instead of trying to fetch one.

| File | Role |
|---|---|
| `Dockerfile` | Image recipe; reads deps from `environment.yml`, installs `jax-fem` at pinned commit `c3fbcb3` |
| `.dockerignore` | Restricts build context to `environment.yml` alone |
| `.gitlab-ci.yml` | Pins the image tag; single `pytest` job on the default branch |
| `ci/build-image.sh` | Builds and tags the image, then prunes build cache |

## Runner host

`slimm-server01.citg.tudelft.nl`, which also hosts three other GitLab runners
(two LaTeX, one myjive) and a local GitLab install.

- Docker root is `/data/docker-root`, on a **49 GB** partition shared with that
  GitLab install's Postgres and Gitaly data. Filling `/data` risks taking the
  GitLab instance down, not just the pipeline — treat disk pressure as urgent.
- `/data` is `/dev/sdb1`, a raw partition, so it cannot be grown in place.
- `concurrent = 1` globally: one job at a time across all four runners.
- A repo clone for building the image lives at `/srv/pdmp-ci`, **owned by the
  user, not root** — root has no GitLab SSH key, so `git pull` must not run
  under `sudo`. `sudo` is needed only for the Docker socket.

## Rebuilding the image

Required whenever `environment.yml`, the `Dockerfile`, or the pinned `jax-fem`
commit changes — historically about once every two months.

```bash
# 1. Commit the change on a branch and push it.
#    `rules:` restricts pipelines to the default branch, so nothing runs yet.

# 2. Build on the runner host from that branch:
ssh slimm-server01.citg.tudelft.nl
cd /srv/pdmp-ci && git fetch && git checkout <branch>
sudo ci/build-image.sh          # tags pdmp-ci:$(date +%Y-%m) and :latest

# 3. Smoke-test before trusting CI:
sudo docker run --rm -v /srv/pdmp-ci:/workspace pdmp-ci:<tag> pytest --cov=pdmp

# 4. Bump `image:` in .gitlab-ci.yml to the new tag, then merge to master.

# 5. Return the clone to master so the next rebuild tracks the right branch:
cd /srv/pdmp-ci && git checkout master && git pull
```

Order matters: the image must exist on the host **before** the tag reaches the
default branch, or the first pipeline fails on a missing image. Building from a
branch first is what makes that possible — the build inputs only exist in the
commit, so "build before pushing" is otherwise circular.

The tag is pinned rather than `latest` so a broken rebuild cannot silently
become what CI runs, and the previous image stays on the host as a rollback.

## Runner configuration

In `/etc/gitlab-runner/config.toml`, under the `pdmp` runner's
`[runners.docker]` block:

```toml
disable_cache = true
pull_policy = "if-not-present"
```

Keep the `image = "continuumio/miniconda3"` fallback: it is unused because
`.gitlab-ci.yml` always specifies an image, but a job that omits `image:` fails
outright without it.

`disable_cache = true` belongs in **all four** runner blocks. It disables only
the Docker-volume-based local cache, not the `cache:` keyword or artifacts —
none of the pipelines on this host use `cache:`, so nothing is lost. Without it
the runner mints a persistent named volume per runner-and-project and never
garbage-collects them; 61 orphans had accumulated across 25 stale runner IDs.

Validate with `sudo gitlab-runner verify` before `sudo gitlab-runner restart` —
a TOML syntax error otherwise leaves all four runners silently dead.

## Housekeeping

Weekly cron on the runner host:

```bash
docker container prune -f && docker image prune -f && docker builder prune -af
```

Note the deliberate absence of `-a` on the image prune. `docker image prune -af`
deletes every image not backing a running container, including
`texlive/texlive:TL2024-historic` (~5–7 GB), which the LaTeX pipelines then
re-pull on their next run — potentially failing if the disk is tight again.

Also set container log caps in `/etc/docker/daemon.json`, then restart Docker:

```json
{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| Job fails pulling `pdmp-ci:<tag>` with a manifest error | `pull_policy = "if-not-present"` not live, or the image was never built on this host |
| `No space left on device` during a job | `/data` full; check `docker system df` and stopped job containers before pruning images |
| `no Docker image specified to run the build in` | `image` missing from both `.gitlab-ci.yml` and the runner's `[runners.docker]` block |
| `git@gitlab.tudelft.nl: Permission denied (publickey)` on the host | Ran git under `sudo`; root has no registered key. Clone and pull as your user |
| `dubious ownership` from git on the host | The `/srv/pdmp-ci` clone is root-owned; `chown` it to your user |
| Tests pass locally, fail in CI | Image is stale — rebuild after any `environment.yml` change |

## Known trade-offs

- **Manual rebuild trigger.** The alternative is a shell-executor runner that
  rebuilds the image in CI automatically, but that requires adding
  `gitlab-runner` to the `docker` group — root-equivalent access on a host
  shared with other people's runners and a GitLab install. Given the rebuild
  cadence (~6×/year), the manual step was judged the better trade.
- **Image size.** `pytorch` + `botorch` + `gpytorch` account for roughly 3 GB
  and appear to be needed only by `tests/test_surrogates_bo.py`. Splitting them
  out would shrink the image noticeably.
