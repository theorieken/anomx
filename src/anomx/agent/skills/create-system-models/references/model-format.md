# Model format

The API stores a model at `System.properties.navigator_appearance`:

| Field | Values | Default |
| --- | --- | --- |
| `shape` | `sphere`, `box`, `cylinder`, `cone`, `torus`, `capsule`, `octahedron`, `plane`, `circle` | `sphere` |
| `color` | Six-digit hexadecimal RGB, e.g. `#80b8d1` | `#8b91ad` |
| `scale` | Number from 0.5 to 2; overall graph size | 1 |
| `metalness` | Number from 0 to 1 | 0.15 |
| `roughness` | Number from 0 to 1 | 0.4 |
| `parts` | Up to 24 component objects | Single `shape` |

A nonempty `parts` array replaces the single primitive. Each component accepts:

| Field | Values |
| --- | --- |
| `name` | Optional component label, at most 80 characters |
| `shape` | A primitive from the table above |
| `color` | Optional six-digit RGB; inherits the model color |
| `metalness`, `roughness` | Optional numbers from 0 to 1; inherit model material |
| `position` | Three model coordinates, each from −4 to 4; default `[0,0,0]` |
| `rotation` | Three Euler XYZ angles in **radians**, each from −6.284 to 6.284; default zero |
| `scale` | Three scale factors, each from 0.1 to 4; default `[1,1,1]` |

The UI shows angles in degrees, but the API stores radians. Y is up. Planes and
circles lie in XY, facing along ±Z; rotate by π/2 around X for a horizontal
surface. Cylinders, cones and capsules extend along Y before rotation.
A box is one model unit along each axis; a cylinder has diameter and height of
one unit. The default sphere diameter is 7/6 units. Component scale factors
multiply those primitive dimensions; these are not engineering measurements.

The complete assembly is centered and normalized to the graph's node footprint.
Overall `scale`, hierarchy size weighting and camera perspective then affect
its displayed size. Changes to this model do not change graph coordinates or
system relationships. An empty parts array falls back to the single shape in
the graph; removing the appearance restores the automatic default.

Example: a schematic vessel, with two supports that remain individually editable:

```json
{
    "color": "#80b8d1",
    "scale": 1.25,
    "metalness": 0.3,
    "roughness": 0.45,
    "parts": [
        {"name": "Vessel", "shape": "capsule", "rotation": [0, 0, 1.5708], "scale": [1, 2, 1]},
        {"name": "Left support", "shape": "box", "color": "#71818e", "position": [-0.7, -0.8, 0], "scale": [0.25, 0.6, 0.7]},
        {"name": "Right support", "shape": "box", "color": "#71818e", "position": [0.7, -0.8, 0], "scale": [0.25, 0.6, 0.7]}
    ]
}
```

Reuse this pattern only when it suits the requested system. Supply actual
numeric arrays and supported fields; the schema does not load scripts, SVG,
HTML, arbitrary geometry, URLs, or external CAD files.
