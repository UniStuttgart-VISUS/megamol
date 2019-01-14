# MegaMol™ Plugin "Cinematic Camera"

This module allows the video rendering of simulations.
By defining fixed keyframes of desired camera positions for specific animation times arbitrary tracking shots can be created.

![cinematic camera demo picture](https://github.com/braunms/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/demo.png)

--- 

## Module Descriptions
This plugin provides the modules `KeyframeKeeper`,  `CinematicRenderer`, `CinematicView`, `TimeLineRenderer` and `ReplacementRenderer`.

#### KeyframeKeeper:

This module maintains the keyframes and their properties centralised. 
The modules `CinematicRenderer`, `CinematicView` and `TimeLineRenderer` have to be connected with the same `KeyframeKeeper` (see module call graph below).

#### CinematicRenderer:

This module shows the spatial position of the defined keyframes and the resulting tracking shot. 
It allows the selection of keyframes by mouse. 
The position and direction of the camera parameters of a keyframe can be altered by selectable manipulators.

#### CinematicView:

This module provides the preview of the final view to be rendered.
Any desired view can be applied to an existing or a new keyframe.
By running the animation as preview one can audit the final view of the video.
The complete final video is defined by the resolution and the number of frames per second and can be rendered to png files.

#### TimeLineRenderer:

This module shows the temporal position of the keyframes on the animation time axis and the simulation time axis in an two-dimensional diagram.
The keyframes can be selected (left mouse button) or they can be shifted along the simulation or animation time axis per drag and drop (right mouse button).
The time axes can be zoomed and shifted independently at the current mouse position (middle mouse button, right mouse button).

#### ReplacementRenderer:

This module offers replacement rendering for models with performance too low for interactivly creating the tracking shot. 
Therefore only the bounding box of the model is drawn.

--- 

## Parameters

The module `KeyframeKeeper` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_applyKeyframe` (Assigned key: `a`): Apply current settings to selected/new keyframe.
* `02_undoChanges` (Assigned key: `CTRL+z`): Undo keyframe changes.
* `03_redoChanges` (Assigned key: `CTRL+y`): Redo keyframe changes.
* `04_deleteKeyframe` (Assigned key: `d`): Deletes the currently selected keyframe.
* `05_maxAnimTime` (`1.0`): The total timespan of the animation.
* `06_snapAnimFrames` (Assigned key: `f`): Snap animation time of all keyframes to fixed frames.
* `07_snapSimFrames` (Assigned key: `g`): Snap simulation time of all keyframes to integer simulation frames.
* `08_linearizeSimTime` (Assigned key: `t`): Linearize simulation time between two keyframes between currently selected keyframe and subsequently selected keyframe.
* `09_interpolTangent` (`0.5`): Length of keyframe tangets affecting curvature of interpolation spline.
* `editSelected - 01_animTime` (`1.0`): Edit animation time of the selected keyframe.
* `editSelected - 02_simTime` (`1.0`): Edit simulation time of the selected keyframe.
* `editSelected - 03_position`: Edit the position vector of the selected keyframe.
* `editSelected - 04_lookat`: Edit the look-at vector of the selected keyframe.
* `editSelected - 05_resetLookat` (Assigned key: `l`): Reset the LookAt vector of the selected keyframe.
* `editSelected - 06_up`:  Edit the up vector of the selected keyframe.
* `editSelected - 07_apertureAngle`: Edit aperture angle of the selected keyframe.
* `storage - 01_filename`:  The name of the file to load or save keyframes. 
* `storage - 02_save` (Assigned key: `CTRL+s`): Save keyframes to file.
* `storage - 03_load` (Assigned key: `CTRL+l`): Load keyframes from file.

The module `CinematicRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_splineSubdivision` (`20`): Amount of interpolation steps between keyframes.
* `02_toggleManipulators` (Assigned key: `m`): Toggle different manipulators for the selected keyframe.            
* `03_toggleHelpText` (Assigned key: `h`): Show/hide help text for key assignments.
* `04_manipOutsideModel` (Assigned key: `w`): Keep manipulators always outside of model bounding box.

Necessary parameter(s) from `View3D`-Module: 

 * `enableMouseSelection` (Assigned Key `tab`): Toggle mouse interaction between scene camera manipulation or keyframe manipulation.

The module `CinematicView` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_renderAnim` (Assigned key: `r`): Toggle rendering of complete animation to png files.   
   Whenever rendering is begun a new folder holding the frame image files (png) is generated.
* `02_playPreview` (Assigned key: `space`): Toggle playing animation as preview.
* `03_skyboxSide` (`NONE`): Select the skybox side.
* `04_cubeMode` (`false`): Activate mode that renders the bounding box side selected with `03_skyboxSide`.
* `05_cinematicWidth` (`1920`): The width resolution of the cinematic view to render.
* `06_cinematicHeight`(`1080`): The height resolution of the cinematic view to render.
* `07_fps` (`24`): The frames per second the animation should be rendered.
* `08_firstRenderFrame` (`0`): Set first frame number to start rendering with.
* `09_delayFirstRenderFrame` (`10.0`): Delay (in seconds) to wait until first frame is ready and rendering to file is started.
* `10_frameFolder` Specify folder where the frame files should be stored.
* `11_addSBSideToName` (`false`): If true, adds the value of `03_skyboxSide` to the filename of the written image.
* `stereo - eye` (`Left`) Eye position (for stereo view).
* `stereo - projection` (`Mono Perspective`) Camera projection.

The module `TimeLineRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_fontSize` (`22.0`): The font size.
* `02_rightFrame` (Assigned key: `right arrow`): Move to right animation time frame.
* `02_leftFrame` (Assigned key: `left arrow`): Move to left animation time frame.
* `04_resetAxes` (Assigned key: `p`): Reset shifted and scaled time axes.

The module `ReplacementRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_replacementRendering` (`false`): Show/hide replacement rendering for the model.
* `02_toggleReplacement` (Assigned key: ` `): Toggle replacement rendering. Key can be assigned via `03_replacmentKeyAssign` parameter. 
* `03_replacmentKeyAssign` (` `): Assign a key for the replacement rendering button `02_toggleReplacement`.
* `04_alpha` (`0.75`): The alpha value of the replacement rendering.
    
---

## How to use the module

The modules should be connected as shown in the module call graph below. 
The dark yellow tagged renderer module can be replaced by any other suitable one. 
The renderer module has to be connected to the `CinematicRenderer` as well as to the `CinematicView`.
The light yellow tagged data source module has to be replaced by a suitable one for the used renderer module.

![megamol example module/call graph](https://github.com/braunms/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/graph.png)

### Example

The sample project files (`cinematic_editor.lua`,`cinematic_keyframes.kf` ) which are supported in the `examples` folder of this plugin have to be copied into the `bin` folder of megamol.
In a shell change to the `bin` folder of the megamol executables and start the program with the command:   
*Under Windows:* `.\mmconsole.exe -p cinematic_editor.lua`   
*Under Linux:* `./megamol.sh -p cinematic_editor.lua`

