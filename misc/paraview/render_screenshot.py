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


# Per-field profiles: everything that differs between the fields lives here; the
# rest of the script (pipeline, camera, render) is shared. Add an entry to plot
# another field.
# camera placements (from ParaView's "save camera placements" trace)
CAMERA_SOL = dict(
    position=[-192.14842531691284, 142.80944970818658, 341.4941951476901],
    focal=[64.48365996525354, 22.09038580528689, 141.90594044693322],
    view_up=[0.2813889523038988, 0.9417480962497558, -0.1842030964215179],
)
CAMERA_E = dict(
    position=[-211.36021954772718, 155.81818285369755, 311.5161180799877],
    focal=[58.17794790993812, 31.922417027821464, 135.20555159200106],
    view_up=[0.2858700168555494, 0.9330011195925636, -0.21860293754208981],
)

# colormap per field: either a path to a ParaView preset JSON (imported, then
# applied) or the name of a ParaView built-in preset (e.g. 'Fast').
CREST_JSON = '/home/leon/Nextcloud/Documents/projects/gradient_samplers/pdmp/misc/paraview/crest.json'

FIELDS = {
    'sol': dict(assoc='POINTS', component='Magnitude', warp=True,
                title=r'$\|u\|$', labels=[0.0, 0.25, 0.5, 0.75, 1.0],
                camera=CAMERA_SOL, cmap=CREST_JSON),
    'E':   dict(assoc='CELLS',  component='',          warp=False,
                title='E',        labels=[0.0, 25, 50, 75, 100],
                camera=CAMERA_E,   cmap='Fast'),   # None -> auto labels
}

# CLI: render.sh [field] [input.vtu] [output.png]. Field defaults to 'sol';
# input/output default to the current dir, so you can cd into any results folder
# and drop the figure there. Output defaults to ./<field>.png to avoid clobbering.
parser = argparse.ArgumentParser(description='Render a field screenshot from a .vtu.')
parser.add_argument('field', nargs='?', default='sol', choices=list(FIELDS),
                    help='field to plot (default: sol)')
parser.add_argument('input', nargs='?', default=None, help='input .vtu (default: ./u.vtu)')
parser.add_argument('output', nargs='?', default=None,
                    help='output .png (default: ./<field>.png)')
args, _ = parser.parse_known_args()

ARRAY = args.field                  # field to color by ('sol' or 'E')
cfg = FIELDS[ARRAY]
INPUT_VTU = args.input or os.path.join(os.getcwd(), 'u.vtu')
OUTPUT_PNG = args.output or os.path.join(os.getcwd(), ARRAY + '.png')

# --- config (edit these) ---
DISPLACEMENT = 'sol'                # point-data vector field used for warping
RESOLUTION = [2000, 1200]           # 2x the window the camera was saved for (1075x554)
ZOOM = 1.0                          # 1.0 = exact saved view; >1 zooms in, <1 zooms out

# warp the geometry by the displacement field to show the deformed shape
WARP_SCALE = 15.0                   # magnification factor for the displacements

# mesh edges drawn over the colored surface ('Surface With Edges')
EDGE_COLOR = '#a6a6a6'             # HTML hex color
LINE_WIDTH = 2.0

# lighting — a fresh batch RenderView can render darker than the GUI. Stock
# ParaView defaults are AMBIENT=0.0, DIFFUSE=1.0, KEY_LIGHT_INTENSITY=0.75.
# Raise KEY_LIGHT_INTENSITY to brighten while keeping the shaded 3D look, or
# raise AMBIENT (and lower DIFFUSE) for flatter, angle-independent brightness.
AMBIENT = 0.0
DIFFUSE = 1.0
KEY_LIGHT_INTENSITY = 1.0

# --- build the pipeline (replaces GetActiveSource/GetActiveView) ---
reader = XMLUnstructuredGridReader(FileName=INPUT_VTU)
reader.UpdatePipeline()

# warp the mesh by the (magnified) displacement field to show the deformed shape
# (only for fields that ask for it, e.g. the displacement itself)
if cfg['warp']:
    warp = WarpByVector(Input=reader)
    warp.Vectors = ['POINTS', DISPLACEMENT]
    warp.ScaleFactor = WARP_SCALE
    warp.UpdatePipeline()
    source = warp
else:
    source = reader

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = RESOLUTION
view.OrientationAxesVisibility = 0   # hide the XYZ orientation indicator

display = Show(source, view)

# colored surface (use 'Surface With Edges' to overlay the mesh wireframe)
display.Representation = 'Surface'
display.EdgeColor = hex_to_rgb(EDGE_COLOR)
display.LineWidth = LINE_WIDTH

# lighting (batch renders can be darker than the GUI)
display.Ambient = AMBIENT
display.Diffuse = DIFFUSE
view.KeyLightIntensity = KEY_LIGHT_INTENSITY

# color by the selected field; vector fields use a component (e.g. Magnitude),
# scalars use a 2-tuple with no component
if cfg['component']:
    ColorBy(display, (cfg['assoc'], ARRAY, cfg['component']))
else:
    ColorBy(display, (cfg['assoc'], ARRAY))

# the GUI does these implicitly; in batch we must ask for them explicitly
display.RescaleTransferFunctionToDataRange(True)

lut = GetColorTransferFunction(ARRAY)

# apply the field's colormap: a preset JSON file (import, then apply its preset)
# or a ParaView built-in preset name (e.g. 'Fast')
cmap = cfg['cmap']
if cmap.endswith('.json'):
    if os.path.exists(cmap):
        ImportPresets(filename=cmap)
        lut.ApplyPreset(json.load(open(cmap))[0]['Name'], True)
    else:
        print('WARNING: colormap file not found, using default:', cmap)
else:
    lut.ApplyPreset(cmap, True)   # built-in preset

# --- scalar bar config ---
bar = GetScalarBar(lut, view)
bar.Title = cfg['title']
bar.ComponentTitle = ''            # or 'Magnitude', 'X', etc.
bar.TitleFontSize = 32
bar.LabelFontSize = 24

# custom ticks when the field defines them, else let ParaView place them
if cfg['labels'] is not None:
    bar.UseCustomLabels = 1
    bar.CustomLabels = cfg['labels']
else:
    bar.UseCustomLabels = 0
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

# explicit, per-field camera placement (from ParaView's "save camera placements"
# trace; replaces ResetCamera so the framing is reproducible).
cam = cfg['camera']

# Zoom by dollying — physically move the camera toward the focal point along the
# view direction. ZOOM>1 brings it closer (object bigger), ZOOM<1 further away.
# (Setting CameraViewAngle directly does NOT work: ParaView re-derives it from
# CameraParallelScale and resets it to 30 on Render, so the trace's zoom is lost.)
position = [f + (p - f) / ZOOM for p, f in zip(cam['position'], cam['focal'])]

view.CameraParallelProjection = 0
view.Set(
    CameraPosition=position,
    CameraFocalPoint=cam['focal'],
    CameraViewUp=cam['view_up'],
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
