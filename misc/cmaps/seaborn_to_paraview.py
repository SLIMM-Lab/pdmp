"""Export a seaborn/matplotlib colormap to a ParaView JSON colormap file.

ParaView's "Choose Preset" dialog (in the Color Map Editor) can import
colormaps from a JSON file via the "Import" button. This script samples a
named colormap and writes it in that format.

Usage:
    python seaborn_to_paraview.py crest
    python seaborn_to_paraview.py crest --samples 256 --output crest.json
    python seaborn_to_paraview.py rocket flare mako --samples 128

Then in ParaView:
    Color Map Editor -> "Choose Preset" -> "Import" -> select the .json file.
"""

import argparse
import json

import seaborn as sns
from matplotlib.colors import Colormap


def colormap_to_paraview(cmap: Colormap, name: str, n_samples: int = 256) -> dict:
    """Sample a matplotlib colormap into a ParaView preset dictionary."""
    rgb_points = []
    for i in range(n_samples):
        x = i / (n_samples - 1)  # scalar value in [0, 1]
        r, g, b, _ = cmap(x)     # discard alpha
        rgb_points.extend([x, r, g, b])

    return {
        "ColorSpace": "RGB",
        "Name": name,
        "RGBPoints": rgb_points,
        "NanColor": [0.0, 0.0, 0.0],
    }


def get_cmap(name: str) -> Colormap:
    """Resolve a colormap by name, trying seaborn palettes then matplotlib.

    A trailing "_r" (e.g. "crest_r") yields the reversed colormap. Both seaborn
    and matplotlib understand the suffix directly; if a particular "_r" variant
    is not registered we fall back to reversing the base colormap ourselves.
    """
    try:
        # seaborn registers crest/flare/rocket/mako/icefire/vlag with matplotlib,
        # but as_cmap=True guarantees we get a Colormap for those palette names.
        return sns.color_palette(name, as_cmap=True)
    except (ValueError, TypeError):
        pass

    import matplotlib.pyplot as plt

    try:
        return plt.get_cmap(name)
    except ValueError:
        if name.endswith("_r"):
            return get_cmap(name[:-2]).reversed()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("colormaps", nargs="+",
                        help="One or more colormap names, e.g. crest rocket flare")
    parser.add_argument("--samples", type=int, default=256,
                        help="Number of control points to sample (default: 256)")
    parser.add_argument("--output", default=None,
                        help="Output .json path (default: <first-name>.json)")
    args = parser.parse_args()

    presets = [colormap_to_paraview(get_cmap(name), name, args.samples)
               for name in args.colormaps]

    output = args.output or f"{args.colormaps[0]}.json"
    with open(output, "w") as f:
        json.dump(presets, f, indent=2)

    print(f"Wrote {len(presets)} colormap(s) to {output}")
    print("Import in ParaView: Color Map Editor -> Choose Preset -> Import")


if __name__ == "__main__":
    main()
