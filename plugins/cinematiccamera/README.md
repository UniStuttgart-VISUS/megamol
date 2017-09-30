# MegaMol plugin "Cinematic Camera"

This module allows the video rendering of simulations.
Through defining fixed keyframes of desired camera positions for specific animation times arbitrary tracking shots can be created.

![cinematic camera demo picture](https://github.com/tobiasrau/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/demo.png)

---

## Use cases

This plugin can be used to render a video of a simulation.

--- 

## Module Descriptions
This plugin provides the modules `KeyframeKeeper`,  `CinematicRenderer`, `CinematicView` and `TimeLineRenderer`.

#### KeyframeKeeper:

This module maintains the keyframes and their properties centralised. 
The module `CinematicRenderer`, `CinematicView` and `TimeLineRenderer` have to be connected with the same `KeyframeKeeper` (see module-call graph below).

#### CinematicRenderer:

This module shows the spatial position of the defined keyframes and the resulting tracking shot. 
It allows the selection of keyframes by mouse. 
The position and direction of the camera parameters of a keyframe can be altered by selectable manipulators.

#### CinematicView:

This module provides the preview of the final view to be rendered.
Any desired view can be applied to an existing or a new keyframe.
By running the animation as preview one can audit the final view of the video in real time.
The complete final video can be rendered to png files by defining the resolution and the number of frames per second.

#### TimeLineRenderer:

This module shows the temporal position of the keyframes on the animation time axis and the simulation time axis in an two-dimensional diagram.
The manipulation of the simulation and the animation time of a keyframe can be done per drag and drop.
The time axes can be zoomed independently at the current mouse position.

--- 

## Parameters

The module `KeyframeKeeper` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_addKeyframe` (Assigned key: `a`): Adds new keyframe at the currently selected time.
* `04_maxAnimTime` (`1.0`): The total timespan of the animation.
* `05_setSameSpeed` (Assigned key: `v`): Move keyframes to get same speed between all keyframes.
* `06_snapAnimationFrames` (Assigned key: `f`): Snap animation time of all keyframes to fixed frames.
* `editSelected - 03_deleteKeyframe` (Assigned key: `d`): Deletes the currently selected keyframe.
* `editSelected - 02_applyView` (Assigned key: `c`): Apply current view to selected keyframe.
* `editSelected - 01_animTime` (`1.0`): Edit animation time of the selected keyframe.
* `editSelected - 02_simTime` (`1.0`): Edit simulation time of the selected keyframe.
* `editSelected - 02_position`: Edit the position vector of the selected keyframe.
* `editSelected - 06_lookat`: Edit the look-at vector of the selected keyframe.
* `editSelected - 07_resetLookat` (Assigned key: `l`): Reset the LookAt vector of the selected keyframe.
* `editSelected - 08_up`:  Edit the up vector of the selected keyframe.
* `editSelected - 09_apertureAngle`: Edit aperture angle of the selected keyframe.
* `storage - 01_filename`:  The name of the file to load or save keyframes. 
* `storage - 02_save` (Assigned key: `s`): Save keyframes to file.
* `storage - 03_autoLoad` (`true`): Load keyframes from file when filename changes.

The module `CinematicRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_splineSubdivision` (`20`): Amount of interpolation steps between keyframes.
* `02_toggleManipulators` (Assigned key: `m`): Toggle between the position manipulators and the look-at and up manipulators of the selected keyframe.            
* `03_toggleHelpText` (Assigned key: `h`): Show/hide help text for key assignments.
* `04_toggleModelBBox` (Assigned key: `t`): Toggle between full rendering of the model and semi-transparent bounding box as placeholder of the model.

The module `CinematicView` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_renderAnim` (Assigned key: `r`): Toggle rendering of complete animation to png files.   
   Whenever rendering is begun a new folder holding the new frame files is generated.
* `02_playPreview` (Assigned key: `space`): Toggle playing animation as preview.
* `03_skyboxSide` (`NONE`): Select the skybox side.
* `04_cinematicHeight`(`1920`): The height resolution of the cinematic view to render.
* `05_cinematicWidth` (`1080`): The width resolution of the cinematic view to render.
* `05_fps` (`24`): The frames per second the animation should be rendered.
    
The module `TimeLineRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `01_fontSize` (`15.0`): The font size.

---

## How to use the module

The modules should be connected as shown in the module-call graph below. 
The dark yellow tagged renderer module can be replaced by any other suitable one. 
The renderer module has to be connected to the `CinematicRenderer` as well as to the `CinematicView`.
The light yellow tagged data source module has to be replaced by a suitable one for the used renderer module.

![megamol example module/call graph](https://github.com/tobiasrau/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/graph.png)

### Example

The sample project supported in the `example` folder of this plugin.
As data source for the PDBLoader (Parameter: pdbFilename) any protein from e.g. the [RCSB](http://www.rcsb.org/pdb/home/home.do) can be used.
In a shell change to the `bin` folder of the megamol executables and start the program with the command:   
` .\mmconsole.exe -p cinematiccam_simplemol.mmprj -i cinematiccamera_simplemol instance`


