## Volume
This plugin provides basic volume rendering functionality.

## Build
This plugin is switched on by default.

## Modules

### RaycastVolumeRenderer

A renderer module that implements a basic, but modern renderer for volume data. Rendering is split into two passes: A compute shader performs volume raycasting and writes the output into a 2D texture. The result is then rendered to the currently bound framebuffer using a screen-filling quad.

The renderer provides the following input slots:

| Slot                | Type                      | Description                                                | Remark   |
|---------------------|---------------------------|------------------------------------------------------------|----------|
| chainRendering      | `CallRender3D`            | Connection to another renderer for chaining                | optional |
| lights              | `CallLight`               | Light sources for the illumination of the scene            | ignored  |
| getData             | `VolumetricDataCall`      | Data source, providing a 3D volume                         |          |
| getTransferFunction | `CallGetTransferFunction` | Transfer function to map volume to color and transparency  |          |

The renderer provides the following parameters:

| Parameter      | Default Value | Description                                                            |
|----------------|---------------|------------------------------------------------------------------------|
| ray step ratio | `1.0`         | Modifies the raycasting step size. Use values below 1.0 to oversample and values greater 1.0 to undersample the volume. |

Example screenshots for the bonsai dat-raw volume dataset using two different transfer functions:

<img src="images/RaycastVolumeRenderer.png" width="49%"> <img src="images/RaycastVolumeRenderer_Fancy.png" width="49%">
