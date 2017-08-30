# MegaMol module/plugin for Cinematic Camera/Tracking Shots / Video Creating
This module allows ...

![random sphere data set with purely ambient occlusion shading] (https://github.com/tobiasrau/megamol-dev/blob/cinematiccamera/plugins/cinematiccamera/demo.png)

## Description of the modules
The plugin provides the modules `KeyframeKeeper`,  `CinematicRenderer`, `CinematicView` and `TimeLineRenderer`.

### The module `KeyframeKeeper` has the following features:

The ...

### The module `CinematicRenderer` has the following features:

The ...

### The module `CinematicView` has the following features:

The ...

### The module `TimeLineRenderer` has the following features:

The ...


## Parameters
The values in brackets indicate the default values:

The module `KeyframeKeeper` exposes the following parameters:
* `01 Add new keyframe`                   (Assigned key: `a`):   Adds new keyframe at the currently selected time.
* `02 Replace selected keyframe`          (Assigned key: `r`):   Replaces selected keyframe at the currently selected time.
* `03 Delete selected keyframe`           (Assigned key: `d`):   Deletes the currently selected keyframe.
* `04 Total time`                         (`1.0`):               The total timespan of the animation.
* `05 Set same Veclocity`                 (Assigned key: `v`):   Move keyframes to get same velocity between all keyframes.
* `Edit Selection - 01 Time`              (`0.0`):               Edit time of the selected keyframe.
* `Edit Selection - 02 Position`:                                Edit position vector of the selected keyframe.
* `Edit Selection - 03 LookAt`:                                  Edit LookAt vector of the selected keyframe.
* `Edit Selection - 04 UP`:                                      Edit Up vector of the selected keyframe.
* `Edit Selection - 05 Aperture`:                                Edit aperture angle of the selected keyframe.
* `Storage - 01 Filename`                 (``):                  The name of the file to load or save keyframes. 
* `Storage - 02 Save Keyframes`           (Assigned key: `s`):   Save keyframes to file.
* `Storage - 03 (Auto) Load Keyframes`    (`true`):              Load keyframes from file when filename changes.

The module `CinematicRenderer` exposes the following parameters:
* `Spline Subdivision`                    (`20`):                Amount of interpolation steps between keyframes.
* `Toggle Manipulator`                    (Assigned key: `m`):   Toggle between position manipulation or lookup manipulation of selected keyframe.            
* `Load Time`                             (`true`):              Load total time of animation from simulation data.
* `enable mouse selection`                (Assigned key: `tab`): Enables mouse selection in `CinematicRenderer`.

The module `CinematicView` exposes the following parameters:
* `Cinematic - 01 Skybox Side`            (`None`):              Select the skybox side rendering.
* `Cinematic - 02 Cinematic Resolution X` (`1920`):              Set resolution of cineamtic view in hotzontal direction."),
* `Cinematic - 03 Cinematic Resolution Y` (`1080`):              Set resolution of cineamtic view in vertical direction"),
    
    
The module `TimeLineRenderer` exposes the following parameters:
* `01 Time Resolution`                    (`10.0`):              The resolution of time on the time line.
* `02 Marker Size`                        (`30.0`):              The the size of the keyframe marker.
* `03 Moving time line`                   (Assigned key: `t`):   Toggle if time line should be moveable while left click of mouse.

## How to use the plugin

The ...
