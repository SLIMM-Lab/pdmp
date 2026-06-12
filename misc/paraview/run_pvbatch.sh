#!/usr/bin/env sh
# Run a ParaView Python script fully headless (no GUI) via the Flatpak pvbatch.
#
# Usage:
#   ./run_pvbatch.sh                 # runs render_screenshot.py next to this file
#   ./run_pvbatch.sh script.py args  # runs an arbitrary script with args
#
# ParaView here is the Flatpak org.paraview.ParaView, sandboxed with
# `filesystems=home`, so the script and all I/O must live under your home dir
# (the sandbox has its own isolated /tmp). The hwloc/PCI lines on stderr are
# harmless flatpak-runtime noise.
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -eq 0 ]; then
    set -- "$DIR/render_screenshot.py"
fi

exec flatpak run --command=pvbatch org.paraview.ParaView "$@"
