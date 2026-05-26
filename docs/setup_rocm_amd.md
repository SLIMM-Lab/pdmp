# JAX + ROCm setup on AMD GPU (Pop!_OS 22.04)

Setup notes for running `pdmp` / `jax-fem` on an AMD GPU via ROCm. Verified target: Radeon RX 9060 XT (RDNA 4, `gfx1200`) on Pop!_OS 22.04 (Ubuntu 22.04 base), kernel 6.17.

Use `environment_amd.yml` (not `environment.yml`) — it pins `jax[rocm7-local]` instead of CUDA JAX.

## Skip `amdgpu-dkms`

AMD's quick-start tells you to install `amdgpu-dkms`. **Don't** — on a recent kernel the in-tree `amdgpu` driver is newer than what DKMS ships and the DKMS build will either fail or conflict. Confirm the in-tree driver is already live:

```bash
ls /dev/kfd /dev/dri/renderD*
```

If both exist, you only need ROCm userspace.

## Install steps

1. Register the ROCm 7.2.3 apt repo:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/7.2.3/ubuntu/jammy/amdgpu-install_7.2.3.70203-1_all.deb
   sudo apt install ./amdgpu-install_7.2.3.70203-1_all.deb
   sudo apt update
   ```

2. Install userspace ROCm without recommendations (so it skips `amdgpu-dkms`):
   ```bash
   sudo apt install python3-setuptools python3-wheel
   sudo apt install --no-install-recommends rocm
   ```
   Installs to `/opt/rocm-7.2.3` with a `/opt/rocm` symlink.

3. Add yourself to GPU groups, then log out / reboot:
   ```bash
   sudo usermod -aG render,video $LOGNAME
   ```

4. Make ROCm libraries discoverable:
   ```bash
   echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
   sudo ldconfig
   ```
   Optional: add `/opt/rocm/bin` to `PATH` in `~/.bashrc` so `rocm-smi` / `rocminfo` are on the path.

5. Verify ROCm sees the GPU:
   ```bash
   /opt/rocm/bin/rocminfo | grep -E 'Name:|gfx'
   ```
   Expect `gfx1200` (or `gfx1201`) for RDNA 4 cards.

6. Create the conda env and install JAX:
   ```bash
   mamba env create -f environment_amd.yml
   conda activate pdmp-jax-amd
   ```
   The `jax[rocm7-local]` extra pulls the JAX ROCm 7 plugin against the system ROCm.

7. Verify JAX sees the GPU:
   ```bash
   python -c "import jax; print(jax.devices())"
   ```
   Expect a `RocmDevice`. If it reports CPU only, try:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=12.0.0
   ```

8. Install `jax-fem` from local clone as in `CLAUDE.md`. Do this only after JAX-on-ROCm is verified, so any failure is attributable.

## Troubleshooting

- **`rocminfo` lists CPU but not GPU**: group membership hasn't taken effect — reboot, then check `groups` includes `render` and `video`.
- **JAX kernel launch errors despite `rocminfo` being happy**: gfx1200 support in ROCm 7 is recent; try `HSA_OVERRIDE_GFX_VERSION=12.0.0` before deeper debugging.
- **`apt install rocm` pulls in `amdgpu-dkms` anyway**: use `--no-install-recommends`, or `apt-mark hold amdgpu-dkms` first.