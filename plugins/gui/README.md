# MegaMol GUI Plugin

This is the plugin providing the GUI for MegaMol.

## Modules

### GUIView

The `GUIView` is a view module which implements a `megamol::core::view::AbstractView`. The `GUIView` provides the complete GUI for MegaMol. Use the `GUIView` module as the main view module of your MegaMol project. 

#### Menu

The menu of the main window of the `GUIView` provides the following options:
(Assigned hotkeys are given in brackets.)

* `File`
    * `Save Project` (`Ctrl + Alt + s`) Save the current project to a file (lua).
    * `Exit` (`Alt + F4`) End the program.
* `Windows` 
    * `Configurator` (`(Shift +) F8`*) Show/Hide the configurator window.
    * `Font Settings` (`(Shift +) F9`*) Show/Hide the font settings window.
    * `Main Windows` (`(Shift +) F10`*) Show/Hide the main window.
    * `Performance Metrics` (`(Shift +) F11`*) Show/Hide the performance metrics window showing the fps or ms.
    * `Transfer Function Editor` (`(Shift +) F12`*)Show/Hide the transfer function editor.
* `Help`: Some information and links concerning the currently running MegaMol.

*Use `Shift` for resetting the window to fit the currrent viewport.

**NOTE:**\
Hotkeys use the key mapping of the US keyboard layout. Other keyboard layouts are currently not considerd or recognised. Consider possible transposed `z` and `y` which are used in `undo` and `redo` hotkeys on text input.

#### Parameters

(The values in brackets indicate the default values.)

* `style` (`Dark Colors`): Color style, theme.
* `state` (` `): Current state of all windows. Automatically updated.

### OverlayRenderer

The `OverlayRenderer` is a rendering module which implements a `megamol::core::view::RendererModule<megamol::core::view::CallRender3D_2>`. The `OverlayRenderer` provides overlay rendering like textures, text, parameter values and transport control icons. Prepend the `OverlayRenderer` module any other existing 3D renderer module you want to have an overlay for. 

#### Parameters

(The values in brackets indicate the default values.)

* `mode` (`Texture`): Overlay mode.
* `anchor` (`Left Top`): Anchor of overlay.
* `position_offset` (`0.0`): Custom relative position offset in respect to selected anchor.
* `texture::file_name` (` `): The file name of the texture.
* `texture::relative_width` (`25.0`): Relative screen space width of texture.
* `transport_ctrl::color` (`0.5f, 0.75f, 0.75f`): Color of transpctrl icons.
* `transport_ctrl::duration` (`3.0`): Duration transport ctrl icons are shown after value changes. Value of zero means showing transport ctrl icons permanently.
* `transport_ctrl::fast_speed` (`5.0`): Define factor of default speed for fast transport ctrl icon threshold.
* `transport_ctrl::value_scaling` (`10.0`): Define factor of default speed for ultra fast transport ctrl icon threshold.
* `transport_ctrl::speed_parameter_name` (` `): The full parameter name for the animation speed, e.g. '::Project_1::View3D_21::anim::speed'.
* `transport_ctrl::time_parameter_name` (` `): The full parameter name for the animation time, e.g. '::Project_1::View3D_21::anim::time'.
* `parameter::prefix` (` `): The parameter value prefix.
* `parameter::sufix` (` `): The parameter value sufix.
* `parameter::name` (` `): The full parameter name, e.g. '::Project_1::View3D_21::cam::position'. Supported parameter types: float, int, Vector2f/3f/4f.
* `label::text` (` `): The displayed text.
* `font::name` (`Roboto Sans`): The font name.
* `font::size` (`20.0`): The font size.
* `font::color` (`0.5f, 0.5f, 0.5f`): The font color.

## Configurator [PROTOTYPE]

The configurator prototype is part of the GUI and can be opened via the GUI menu: `Windows`/`Configurator`.\
Use the example project: `-p ../examples/configurator.lua` for starting the configurator automatically:

        mmCreateView("testspheres", "GUIView", "::gui")
        mmSetParamValue("::gui::autostart_configurator", "true")

#### Menu

The menu of the configurator window provides the following options:
(Assigned hotkeys are given in brackets.)

* `File`
    * `Load` 
        * `New` Load new empty project containing only a `GUIView` as starting point.
        * `File` Load existing project from a file.
        * `Running` Load currently running project.
    * `Add` 
        * `File` Add existing project from a file to currently selected project.
        * `Running` Add currently running project to currently selected project.
    * `Save Project` (` `) Save the project of the currently selected tab to a file (lua).
* `[?]` Information on additionally available options.

**NOTE:**\
Projects setting parameter values using `mmSetParamValue` must have the parameter values enclosed in `[=[`and `]=]` delimiters. String delimiters `"` for parameter values are deprecated.

![configurator demo picture](configurator.png)

## Plugin Class Dependencies

![gui plugin class dependencies](class_dependencies.png)
