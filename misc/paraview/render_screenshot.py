"""Headless ParaView screenshot — no GUI window ever opens.

Standalone port of save_screenshot.py (which is meant to be pasted into the
interactive ParaView Python shell). This version builds the pipeline itself
instead of grabbing GetActiveView()/GetActiveSource() from a live session, so
it can run under pvbatch.

Run it via the wrapper (recommended):

    ./run_pvbatch.sh                       # runs this script on the example

or directly:

    flatpak run --command=pvbatch org.paraview.ParaView render_screenshot.py

Note: ParaView is a Flatpak here, sandboxed with `filesystems=home`. Keep the
script and all input/output paths under /home/leon — the sandbox has its own
isolated /tmp. Edit the constants below to point at a different mesh/output.
"""

import argparse
import json
import os

import paraview
from paraview.simple import *

# Stop ParaView from auto-fitting the camera on the first render (ResetCamera on
# Show). Without this, our explicit camera placement below is silently clobbered
# every run — the image gets reframed to fit-all-data instead of the saved view.
paraview.simple._DisableFirstRenderCameraReset()


def hex_to_rgb(s):
    """'#rrggbb' -> [r, g, b] floats in [0, 1] (ParaView's color format)."""
    s = s.lstrip('#')
    return [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# Input/output default to the current working directory, so you can cd into any
# results folder and drop the figure there. Override via positional CLI args:
#   render.sh [input.vtu] [output.png]
parser = argparse.ArgumentParser(description='Render a warped-displacement screenshot.')
parser.add_argument('input', nargs='?', default=os.path.join(os.getcwd(), 'u.vtu'),
                    help='input .vtu (default: ./u.vtu)')
parser.add_argument('output', nargs='?', default=os.path.join(os.getcwd(), 'figure.png'),
                    help='output .png (default: ./figure.png)')
args, _ = parser.parse_known_args()

# --- config (edit these) ---
INPUT_VTU = args.input
OUTPUT_PNG = args.output
ARRAY = 'sol'                       # point-data (vector) array: displacement + coloring
RESOLUTION = [2000, 1200]           # 2x the window the camera was saved for (1075x554)
ZOOM = 1.0                          # 1.0 = exact saved view; >1 zooms in, <1 zooms out

# warp the geometry by the displacement field to show the deformed shape
WARP_SCALE = 15.0                   # magnification factor for the displacements

# mesh edges drawn over the colored surface ('Surface With Edges')
EDGE_COLOR = '#a6a6a6'             # HTML hex color
LINE_WIDTH = 2.0

# custom colorbar: a ParaView preset JSON (export from ParaView's color map editor)
CMAP_FILE = '/home/leon/Nextcloud/Documents/projects/gradient_samplers/pdmp/misc/paraview/crest.json'
CMAP_PRESET = ''                    # preset name to apply; '' = use the first one in CMAP_FILE

# --- build the pipeline (replaces GetActiveSource/GetActiveView) ---
reader = XMLUnstructuredGridReader(FileName=INPUT_VTU)
reader.UpdatePipeline()

# warp the mesh by the (magnified) displacement field to show the deformed shape
warp = WarpByVector(Input=reader)
warp.Vectors = ['POINTS', ARRAY]
warp.ScaleFactor = WARP_SCALE
warp.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = RESOLUTION
view.OrientationAxesVisibility = 0   # hide the XYZ orientation indicator

display = Show(warp, view)

# colored surface + mesh edges (wireframe) in a custom color
display.Representation = 'Surface With Edges'
display.EdgeColor = hex_to_rgb(EDGE_COLOR)
display.LineWidth = LINE_WIDTH

# color by magnitude of the vector field 'sol'
ColorBy(display, ('POINTS', ARRAY, 'Magnitude'))
# use ('CELLS', ARRAY, 'Magnitude') if it is a cell array

# the GUI does these implicitly; in batch we must ask for them explicitly
display.RescaleTransferFunctionToDataRange(True)

lut = GetColorTransferFunction(ARRAY)

# load a custom colorscheme from a ParaView preset JSON and apply it
if os.path.exists(CMAP_FILE):
    ImportPresets(filename=CMAP_FILE)
    preset = CMAP_PRESET or json.load(open(CMAP_FILE))[0]['Name']
    lut.ApplyPreset(preset, True)
else:
    print('WARNING: colormap file not found, using default:', CMAP_FILE)

# --- scalar bar config ---
bar = GetScalarBar(lut, view)
bar.Title = r'$\|u\|$'
bar.ComponentTitle = ''            # or 'Magnitude', 'X', etc.
bar.TitleFontSize = 32
bar.LabelFontSize = 24

# force the intermediate ticks
bar.UseCustomLabels = 1
bar.CustomLabels = [0.0, 0.25, 0.5, 0.75, 1.0]
bar.AddRangeLabels = 1             # keep min/max in addition to custom ticks
bar.DrawTickMarks = 1
bar.DrawTickLabels = 1

# geometry as fractions of window
bar.ScalarBarThickness = 25        # in points
bar.Orientation = 'Vertical'
bar.WindowLocation = 'Any Location'
bar.Position = [0.90, 0.25]           # [x, y] fractions of window, bottom-left of the bar
bar.ScalarBarLength = 0.5             # fraction of window height

display.SetScalarBarVisibility(view, True)

# explicit camera placement (copied from ParaView's "save camera placements"
# trace; replaces ResetCamera so the framing is reproducible).
# Values from the trace:
# CAMERA_POSITION = [-188.67614855196095, 139.89145301646425, 347.65069706383014]
# CAMERA_FOCAL_POINT = [67.9559367302056, 24.172389113564552, 148.06244236307336]
# CAMERA_VIEW_UP = [0.2813889523038988, 0.9417480962497558, -0.1842030964215179]

CAMERA_POSITION = [-192.14842531691284, 142.80944970818658, 341.4941951476901]
CAMERA_FOCAL_POINT = [64.48365996525354, 22.09038580528689, 141.90594044693322]
CAMERA_VIEW_UP = [0.2813889523038988, 0.9417480962497558, -0.1842030964215179]

# Zoom by dollying — physically move the camera toward the focal point along the
# view direction. ZOOM>1 brings it closer (object bigger), ZOOM<1 further away.
# (Setting CameraViewAngle directly does NOT work: ParaView re-derives it from
# CameraParallelScale and resets it to 30 on Render, so the trace's zoom is lost.)
position = [f + (p - f) / ZOOM for p, f in zip(CAMERA_POSITION, CAMERA_FOCAL_POINT)]

view.CameraParallelProjection = 0
view.Set(
    CameraPosition=position,
    CameraFocalPoint=CAMERA_FOCAL_POINT,
    CameraViewUp=CAMERA_VIEW_UP,
)
Render(view)

# --- save ---
SaveScreenshot(
    OUTPUT_PNG,
    view,
    ImageResolution=RESOLUTION,
    TransparentBackground=0,
    OverrideColorPalette='WhiteBackground',  # omit if you want current palette
)
print('Wrote', OUTPUT_PNG)
