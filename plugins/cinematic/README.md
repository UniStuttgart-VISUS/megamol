# MegaMol Plugin: Cinematic

This plugin allows the video rendering (separate file per frame) of any rendering output in MegaMol.
By defining fixed keyframes for desired camera positions and specific animation times, arbitrary tracking shots can be created.

![cinematic demo picture](demo.png)

<!-- TOC -->
## Contents
- [Module Descriptions](#module-descriptions) 
   - [KeyframeKeeper](#keyframekeeper)  
   - [TrackingShotRenderer](#trackingshotrenderer)  
   - [CinematicView](#cinematicview)  
   - [TimeLineRenderer](#timelinerenderer)  
   - [ReplacementRenderer](#replacementrenderer)  
- [Usage](#usage)  
   - [Example](#example)  
   - [How to use the cinematic editor](#How-to-use-the-cinematic-editor)  
<!-- /TOC -->
--- 

## Module Descriptions
This plugin provides the modules `KeyframeKeeper`,  `TrackingShotRenderer`, `CinematicView`, `TimeLineRenderer` and `ReplacementRenderer`.

#### KeyframeKeeper

This module maintains the keyframes and their properties centralised. 
The modules `TrackingShotRenderer`, `CinematicView` and `TimeLineRenderer` have to be connected with the same `KeyframeKeeper` (see module call graph below).

The module `KeyframeKeeper` exposes the following parameters:
(The values in brackets indicate the default values.)

* `applyKeyframe` (Assigned key: `SHIFT + a`): Apply current settings to selected or new keyframe.
* `undoChanges` (Assigned key: `SHIFT + z`): Undo keyframe changes.
* `redoChanges` (Assigned key: `SHIFT + y`): Redo keyframe changes.
* `deleteKeyframe` (Assigned key: `SHIFT + d`): Deletes the currently selected keyframe.
* `maxAnimTime` (`1.0`): The total timespan of the animation.
* `snapAnimFrames` (Assigned key: `SHIFT + f`): Snap animation time of all keyframes to fixed frames.
* `snapSimFrames` (Assigned key: `SHIFT + g`): Snap simulation time of all keyframes to integer simulation frames.
* `linearizeSimTime` (Assigned key: `SHIFT + t`): Linearize simulation time between two keyframes between currently selected keyframe and subsequently selected keyframe.
* `interpolTangent` (`0.5`): Length of keyframe tangets affecting curvature of interpolation spline.
* `editSelected::animTime` (`1.0`): Edit animation time of the selected keyframe.
* `editSelected::simTime` (`1.0`): Edit simulation time of the selected keyframe.
* `editSelected::positionVector`: Edit the position vector of the selected keyframe.
* `editSelected::lookAtVector`: Edit the 'look at' vector of the selected keyframe.
* `editSelected::resetLookAt` (Assigned key: `SHIFT + l`): Reset the 'look at' vector of the selected keyframe to the center of the model boundng box.
* `editSelected::upVector`:  Edit up vector direction relative to 'look at' vector of the selected keyframe.
* `editSelected::apertureAngle`: Edit aperture angle of the selected keyframe.
* `storage::filename`:  The name of the file to load or save keyframes. 
* `storage::save` (Assigned key: `SHIFT + s`): Save keyframes to file.
* `storage::load` (Assigned key: `SHIFT + l`): Load keyframes from file.

#### TrackingShotRenderer

This module shows the spatial position of the defined keyframes and the resulting tracking shot. 
It allows the direct selection of keyframes by left mouse button. 
The position and direction of the camera parameters of a keyframe can be altered by drag & drop with left mouse button on shown manipulators.
There are two modes for showing differnt manipulators for a selected keyframe.

The module `TrackingShotRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `splineSubdivision` (`20`): Amount of interpolation steps between keyframes.          
* `helpText` (Assigned key: `SHIFT + h`): Show/hide help text for key assignments.
* `manipulators::toggleVisibleGroup` (Assigned key: `SHIFT + q`): Toggle visibility of different manipulator groups.  
* `manipulators::visibleGroup`: Select visible manipulator group.  
* `manipulators::showOutsideBBox` (Assigned key: `SHIFT + w`): Show manipulators always outside of model bounding box.

#### CinematicView

This module provides the preview of the final view to be rendered.
Here a desired camera view can be chosen.
The desired view can be applied to an existing selected keyframe or a new keyframe.
By running the animation as preview one can audit the final view of the video.
The complete final video is defined by the resolution and the number of frames per second and can be rendered to png files.

The module `CinematicView` exposes the following parameters:
(The values in brackets indicate the default values.)

* `cinematic::renderAnim` (Assigned key: `SHIFT + r`): Toggle rendering of complete animation to png files.   
   Whenever rendering is begun a new folder holding the frame image files (png) is generated.
* `cinematic::playPreview` (Assigned key: `SHIFT + Space`): Toggle playing animation as preview.
* `cinematic::skyboxSide` (`NONE`): Select the skybox side.
* `cinematic::cubeMode` (`false`): Activate mode that renders the bounding box side selected with `skyboxSide`.
* `cinematic::cinematicWidth` (`1920`): The width resolution of the cinematic view to render.
* `cinematic::cinematicHeight`(`1080`): The height resolution of the cinematic view to render.
* `cinematic::fps` (`24`): The frames per second the animation should be rendered.
* `cinematic::firstRender` (`0`): Set first frame number to start rendering with.
* `cinematic::lastRender` (): Set last frame number to end rendering with.
* `cinematic::delayFirstFrame` (`10.0`): Delay (in seconds) to wait until first frame is ready and rendering to file is started.
* `cinematic::frameFolder` Specify folder where the frame files should be stored.
* `cinematic::addSBSideToName` (`false`): If true, adds the value of `skyboxSide` to the filename of the written image.
* `cinematic::stereo_eye` (`Left`) Eye position (for stereo view).
* `cinematic::stereo_projection` (`Mono Perspective`) Camera projection.

#### TimeLineRenderer

This module shows the temporal position of the keyframes on the animation time axis and the simulation time axis in an two-dimensional diagram.
The keyframes can be selected (left mouse button) or they can be shifted along the simulation or animation time axis per drag and drop (right mouse button).
The time axes can be zoomed (middle mouse button) and shifted (right mouse button) independently at the current mouse position.

The module `TimeLineRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `gotoRightFrame` (Assigned key: `right arrow`): Move to right animation time frame.
* `gotoLeftFrame` (Assigned key: `left arrow`): Move to left animation time frame.
* `resetAxes` (Assigned key: `SHIFT + x`): Reset all shifted and scaled time axes.

#### ReplacementRenderer

This module offers replacement rendering for models with performance too low for interactivly creating the tracking shot. 
Therefore only a coloured bounding box of the model is drawn.

The module `ReplacementRenderer` exposes the following parameters:
(The values in brackets indicate the default values.)

* `alpha` (`0.75`): The alpha value of the replacement rendering.
* `hotkeyAssignment` (` `): Choose hotkey for replacement rendering button `toggleReplacement`.
* `toggleReplacement` (Assigned key: ` `): Toggle replacement rendering. Hotkey can be chosen from list offered by `hotkeyAssignment` parameter. 
* `replacement` (`false`): Show/hide replacement rendering for chained renderer (coloured bounding box).

---

## Usage

The modules for a complete cinematic editor should be connected as shown in the *Cinematic Editor* module group below. 
The renderer module and the data source module in the *External Project* group can be replaced by any other suitable modules. 

**Note:** The renderer module has to be connected to the `TrackingShotRenderer` as well as to the `CinematicView`.
For simplification the preferred way of adding the cinematic graph to a new project is to use the predefined cinematic editor project `examples/cinematic/cinematic_editor_megamol.lua` project (see example below).

![megamol example module call graph](graph.png)

### Example

In order to run the example change to the `bin` folder of the megamol executable in a (bash/powershell) shell and start the program with the following command:

*Under Windows:* `.\megamol.exe ../examples/cinematic/cinematic_editor_megamol.lua`

*Under Linux:* `./megamol ../examples/cinematic/cinematic_editor_megamol.lua`

### How to use the cinematic editor

In the `cinematic_editor_megamol.lua` project file, the test sphere project `examples/testsphere_megamol.lua` is automatically appended to the cinematic module graph (see *External Project* module group in graph above). 
Additionally the corresponding keyframe file for the testsphere project `examples/cinematic/cinematic_keyframes.kf` is loaded. 
Any other `lua` project file can be included by changing the variable `project_file` to the projects file path you want to use.
The keyframe file can be set to the empty string `""` for the initial loading of a new project and can be set to a newly created keyframe file later.
