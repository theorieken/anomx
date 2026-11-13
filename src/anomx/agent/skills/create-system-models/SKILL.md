---
name: create-system-models
description: Create, edit, or reuse a system's compound 3D representation for Anomx graphs and the system model editor. Use for visual models, not system hierarchy, icons, simulation meshes, or manufacturing CAD exports.
metadata:
    title: Create System Models
    requires_platform: true
    keywords: system, 3d, model, geometry, visualization, mini cad, design
    anomx_models: systems_system
---

# Create system models

Design a recognizable, compact system representation using the same primitives
as the platform's **System → Edit → Editor** tab. Read `use-anomx-api` for the
connected API conventions and [the model format](references/model-format.md)
before preparing a design. The editor and the 3D graphs consume the same
`properties.navigator_appearance` value.

## Resolve and inspect

Find the requested system through `GET /systems?query=<name>&limit=20`, then read
`GET /systems/<id>` before writing. Establish which physical or functional form
the user wants. Use supplied references and the system description; do not
invent dimensions, machinery details, or a health state from a name alone.
For an unspecified appearance, make a simple schematic silhouette and describe
it as such. Preserve an existing custom design unless the request covers editing
or replacing it.

To reuse a design, browse
`GET /systems?has_custom_visualization=true&query=<name>&limit=20&ordering=name`
and follow the returned pagination. Read the chosen source's properties. Copy
only its appearance into the target; names, metadata, hierarchy, and object
permissions belong to their original systems. If the server does not support
this filter or the requested primitive, inspect its schema and report the
capability mismatch instead of silently claiming the model was saved.

## Design

Start with the fewest primitives that make the silhouette recognizable at node
size. Use named components for meaningful pieces so the user can select and
edit them later. Position, rotate and scale components to assemble the model;
use colors and material values deliberately. A cylinder with supports often
communicates a vessel better than many tiny decorative components.

The graph centers and normalizes the complete assembly; coordinates and the
editor's dimensions are **model units**, not physical measurements. Check the
front, side, and top arrangement when preparing compound shapes. Planes and
circles are flat, two-sided surfaces; use boxes or cylinders for thickness.
Keep all values finite and within the format's bounds. The representation is
declarative data, not executable code or an external mesh URL.

## Save and verify

Refresh the target before writing. Merge the prepared appearance into its
current `properties`, preserving every unrelated key, especially `icon` and
existing metadata. `PATCH /systems/<id>` replaces the whole properties object;
send a `properties` field containing that merged object. Use the
normal platform API and the user's authorized scope. Appearance changes do not
authorize creating systems, changing roles or connections, editing metadata,
or overwriting a different system's design. Respect write denials.

Read the saved target back and compare the returned appearance to the intended
design. If a visual preview is available, inspect the silhouette and component
overlap in the editor or graph. Distinguish readback verification from actual
visual verification. Report the affected system and what changed.

To restore automatic branch coloring and the default sphere, remove only
`navigator_appearance` from the merged properties. Do not replace the entire
properties object with an empty object.
