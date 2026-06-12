#!/usr/bin/env sh
# Render a warped-displacement screenshot from ANY directory, headless.
#
# Usage (from the folder containing your u.vtu):
#   /abs/path/to/render.sh                       # reads ./u.vtu, writes ./figure.png
#   /abs/path/to/render.sh in.vtu out.png        # explicit input/output
#
# Defaults are resolved relative to your current directory (the flatpak sandbox
# preserves the host CWD), so the figure lands wherever you run this from. Put
# this on your PATH for `render.sh` from anywhere. Everything must be under
# /home/leon — the flatpak only mounts the home dir.
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"

exec flatpak run --command=pvbatch org.paraview.ParaView \
    "$DIR/render_screenshot.py" "$@"
