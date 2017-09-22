# MegaMol plugin "Cinematic Camera"

This module allows the creation of arbitary tracking shots by defining keyframes to render videos of simulations.

![cinematic camera demo picture](https://github.com/tobiasrau/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/demo.png)

## Description of the modules
The plugin provides the modules `KeyframeKeeper`,  `CinematicRenderer`, `CinematicView` and `TimeLineRenderer`.

#### Description of the module `KeyframeKeeper`:

This module maintains the keyframes centralised. The module `CinematicRenderer`, `CinematicView` and `TimeLineRenderer` have to be connected with the same `KeyframeKeeper`.

#### Description of the module `CinematicRenderer`:

This module shows the tracking shot of the defined keyframes. It allows selecting keyframes by mouse. The position and direction of the camera can be altered by selectable manipulators.

#### Description of the module `CinematicView`:

This module provides the preview of the final view.

#### Description of the module `TimeLineRenderer`:

This module offers manipulation of the keyframes on the simulation time axis and the animation time axis.


## Parameters

The module `KeyframeKeeper` exposes the following parameters:
(The values in brackets indicate the default values.)
* `01_addKeyframe` (Assigned key: `a`): Adds new keyframe at the currently selected time.
* `04_maxAnimTime` (`1.0`): The total timespan of the animation.
* `05_setSameSpeed` (Assigned key: `v`): Move keyframes to get same speed between all keyframes.
* `06_snapAnimationFrames` (Assigned key: `f`): Snap animation time of all keyframes to fixed animation frames.
* `editSelected - 03_deleteKeyframe` (Assigned key: `d`): Deletes the currently selected keyframe.
* `editSelected - 02_applyView` (Assigned key: `c`): Apply current view to selected keyframe.
* `editSelected - 01_animTime` (`1.0`): Edit animation time of the selected keyframe.
* `editSelected - 02_simTime` (`1.0`): Edit simulation time of the selected keyframe.
* `editSelected - 02_position`: Edit  position vector of the selected keyframe.
* `editSelected - 06_lookat`: Edit LookAt vector of the selected keyframe.
* `editSelected - 07_resetLookat` (Assigned key: `l`): Reset the LookAt vector of the selected keyframe.
* `editSelected - 08_up`:  Edit Up vector of the selected keyframe.
* `editSelected - 09_apertureAngle`: Edit apperture angle of the selected keyframe.
* `storage - 01_filename` (``):  The name of the file to load or save keyframes. 
* `storage - 02_save` (Assigned key: `s`): Save keyframes to file.
* `storage - 03_autoLoad` (`true`): Load keyframes from file when filename changes.

The module `CinematicRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)
* `01_splineSubdivision` (`20`): Amount of interpolation steps between keyframes.
* `02_toggleManipulators` (Assigned key: `m`): Toggle between position manipulators and lookat/up manipulators of selected keyframe.            
* `03_toggleHelpText` (Assigned key: `h`): Show/hide help text for key assignments.

The module `CinematicView` exposes the following parameters:
(The values in brackets indicate the default values.)
* `01_renderAnim` (Assigned key: `r`): Toggle rendering of complete animation to PNG files. New folder for frames is generated in the format: "frames_<fps>fps_yyyymmdd_hhmmss"
* `02_playPreview` (Assigned key: `space`): Toggle playing animation as preview.
* `03_skyboxSide` (`NONE`): Select the skybox side.
* `04_cinematicHeight`(`1920`): The height resolution of the cineamtic view to render.
* `05_cinematicWidth` (`1080`): The width resolution of the cineamtic view to render.
* `05_fps` (`24`): Frames per second the animation should be rendered.
    
The module `TimeLineRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)
* `01_fonSize` (`15.0`): The font size.


## How to use the plugin

The modules should be connected as shown in the following module/call graph. The dark yellow tagged render module can be replaced by any other suitable render module. The light yellow tagged data source module has to be replaced by an suitable data source for the used render module.


![megamol example module/call graph](https://github.com/tobiasrau/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/graph.png)

Example project files are supported in the `\example` fodler of this plugin.


