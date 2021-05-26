# MegaMol Plugin: GUI

This is the plugin that provides the GUI for MegaMol.

![gui](gui.jpg)

<!-- TOC -->
## Contents
- [Graphical User Interface (GUI)](#graphical-user-interface-gui) 
    - [Main Menu](#main-menu) 
    - [Configurator](#configurator) 
        - [Main Menu](#main-menu) 
        - [*Module Stock List* Sidebar](#module-stock-list-sidebar) 
        - [*Module Parameters* Sidebar](#module-parameters-sidebar) 
        - [Project Tab ](#project-tab) 
            - [Menu Bar](#menu-bar) 
            - [Graph](#graph) 
            - [Module Groups](#module-groups) 
            - [Module Group Interface Slots](#module-group-interface-slots) 
    - [Hotkey Overview](#hotkey-overview)             
- [Modules](#modules) 
    - [GUIView](#guiview) 
    - [OverlayRenderer](#overlayrenderer) 
        - [Parameters](#parameters) 
- [Information for Developers](#information-for-developers) 
    - [Using ImGui in Modules](#using-imgui-in-modules) 
    - [New Parameter Widgets](#new-parameter-widgets) 
        - [How to add a new parameter widget](#how-to-add-a-new-parameter-widget) 
        - [How to add a new parameter group widget](#how-to-add-a-new-parameter-group-widget) 
    - [Default GUI State ](#default-gui-state) 
    - [Graph Data Structure](#graph-data-structure) 
    - [Class Dependencies](#class-dependencies) 
<!-- /TOC -->

---

## Graphical User Interface (GUI)

The GUI of MegaMol is based on [Dear ImGui](https://github.com/ocornut/imgui) *Version 1.82 (Docking Branch)*.  

See the *Bug and Feature Tracker* [#539](https://github.com/UniStuttgart-VISUS/megamol/issues/539) for current work in progress. 

**DISCLAIMER: The following descriptions are no longer valid for the old frontend *mmconsole***   

**NOTE**
* Hotkeys use the key mapping of the US keyboard layout. Other keyboard layouts are currently not considered or recognized. Consider possible transposed `z` and `y` which are used in `undo` and `redo` hotkeys on text input.
* Parameter values in the lua command `mmSetParamValue` must have the value enclosed in `[=[`and `]=]` delimiters. String delimiters `"` for parameter values are not supported any more.

### Main Menu

The main menu provides the following options:
(Assigned hotkeys are given in brackets.)

* **`File`**
    * **`Load Project`** (`Ctrl + l`) Load project from file (lua).
    * **`Save Project`** (`Ctrl + s`) Save the current project to file (lua). *If you want to save the current **GUI state**, check the option `Save GUI state` in the file dialog!* (corresponding LUA option: `mmSetGUIState(>JSON String>)/mmgetGUIState()`)
    * **`Exit`** (`Alt + F4`) End the program.
* **`Windows`** 
    * **`Menu`** (`F12`) Show/Hide the menu.
    * **`Configurator`** (`F11`) Show/Hide the configurator window. See detailed description below.
    * **`Parameters`** (`F10`) Show/Hide the parameter window. *In order to enable more options to change the parameters appearance (widget), you can switch the `Mode` from **Basic** to **Expert**.*
    * **`Log Console`** (`F9`) Show/Hide the log console window. In order to prevent automatically showing the log console on errors and warning, disable the `Force Open` option.
    * **`Transfer Function Editor`** (`F8`) Show/Hide the transfer function editor. You have to `Apply` changes in order to take effect.
    * **`Performance Metrics`** (`F7`) Show/Hide the performance metrics window showing the fps or ms. 
    * **`Show/Hide All Windows`** (`Ctrl + g`) Show/Hide all GUI windows (corresponding LUA/CLI option: `mmSetGUIVisible(<bool>)/mmGetGUIVisible()`/ `guishow`).
    * **`Windows Docking Preset`** Load the docking preset for all windows.
* **`Screenshot`** 
    * **`Select Filename`** (`megamol_screenshot.png`) Open file dialog for choosing file name for screenshot. Current filename is shown in menu. After each screenshot triggering a incrementing suffix (`_1`) is appended to the initially given filename. This way the file name does not have to be changed for each screenshot.
    * **`Trigger`** (`F2`) Trigger screenshot.
* **`Projects`** 
    * *`<Names of currently opened projects>`* Click circle button to run stopped project.
      * **`Toggle Graph Entry`** (`F3`) Toggle between available graph entries. After toggling graph entry, intermediate state disables all graph entries.
* **`Settings`**    
    * **`Style`** Select predefined GUI Style (ImGui Dark Colors, ImGui Light Colors, Corporate Gray, Corporate White).
    * **`Font`** Select predefined GUI Style.
        * `Select Available Font` Choose among preloaded fonts.
        * `Load font from file` Load new font from TTF-file (se provided fonts in /share/resources/fonts/) in any desired size.  
    * **`Scale`** Select fixed GUI scaling (100%, 150%, 200%, 250%, 300%). Support for high DPI displays. (corresponding LUA/CLI option: `mmSetGUIScale(<float>)/mmGetGUIScale()` / `guiscale`)
* **`Help`**
    * **`About`**: Some information and links concerning the currently running MegaMol.

### Configurator

The project configurator is part of the GUI and can be opened via the main menu: `Windows`/`Configurator`.  

![configurator demo picture](configurator.png)

#### Main Menu

* **`File`**
    * **`New Project`** Create new empty project.
    * **`Open Project`** Open existing project in new tab.
    * **`Add Project`** Open existing project and add to currently selected project.
    * **`Save Project`** (`Ctrl + Shift + s`) Save the project of the currently selected tab to file (.lua)
* **`View`**
    * **`Modules Sidebar`** Show/Hide sidebar with module stock list.
    * **`Parameter Sidebar`** Show/Hide sidebar with parameters of currently selected module.

#### *Module Stock List* Sidebar

* Search for Module (`Ctrl + Shift + m`)
* Add Module from Stock List to Graph
    * `Double Left Click` on Module in Stock List
    * `Right Click` on Module in Stock List - Context Menu: Add
    
#### *Module Parameters* Sidebar

* Search for Parameter (`Ctrl + Shift + p`)
        
#### Project Tab 

##### Menu Bar

* `Main View` Change main view state of currently selected view module.
* `Reset Scrolling` Reset scrolling.
* `Reset Zooming` Reset zooming.
* `Grid` Show/Hide grid.
* `Call Names` Show/Hide call names.
* `Module Names` Show/Hide module names and other decoration.
* `Slot Names` Show/Hide slot names.
* `Layout Graph` Apply simple layout of the project graph.

##### Graph *(= Node-Link-Diagram)*

* Call Creation
    * Drag and Drop from Call Slot to another highlighted compatible Call Slot.
* Add Call Slot to existing Interface Slot
    * Drag and Drop from Call Slot to another highlighted compatible Interface Slot. 
* Spawn *Module Stock List* in pop-up window at mouse position.
    * `Ctrl + Shift + m`
    * `Double Left Click`    
* Sidebar Splitter Collapsing/Expansion
    * `Double Right Click` on Splitter
* Graph Zooming
    * `Mouse Wheel`
* Graph Scrolling
    * `Middle Mouse Button`
* Multiple Selection of Modules
    * Drag & Drop with `Left Mouse Button` (start outside of any module). Each module partially overlapped by the frame will be selected.
    * Hold `Shift` + `Left Click` on modules you additionally want to select/deselect to/from current selection.
* `Module`
    * Main View `Radio Button`: Toggle main view flag (only available for view modules).
    * Parameter `Arrow Button`: Show/Hide *Module Parameters* in small window stuck to the module.
    * Context Menu (`Right Click`)
        * Delete (Alternative: Select with `Left Click` an press `Delete`)
        * Layout (Only available when multiple modules are selected)
        * Rename
        * Add to Group
            * New
            * *Select from list of existing groups*
        * Remove from Group
* `Call`
    * Context Menu (`Right Click`)
        * Delete (Alternative: Select with `Left Click` an press `Delete`)
* `Call Slot` 
    * `Double Left Click` Show *Module Stock List* in pop-up window.
    * Context Menu (`Right Click`)    
        * Create new Interface Slot
        * Remove from Interface Slot
* `Group`(-Header)
    * `Double Left Click` Toggle between expanded and collapsed view.
    * Context Menu (`Right Click`)
        * Collapse View / Expand View
        * Layout 
        * Rename      
        * Delete (Alternative: Select with `Left Click` an press `Delete`)

##### Module Groups

Modules can be bundled in groups. 
This allows a minimized (collapsed) view of large module groups inside the configurator. 
Grouped modules can be reused when stored separately. 
Module groups can be added to any other projects.
Module groups are stored in project files using the already available module namespace (which is currently unused in the core). 

##### Module Group Interface Slots

Call slots of modules, which are part of a group, can be added to the group interface. 
This generates new interface slot which allow outgoing calls. 
Interface slots are automatically generated if required.
Additionally, call slots can be added to the groups interface via the context menu of the call slots.
Caller interface slots (on the right side of a group) allow the bundling of more than one connection to the same callee slot of another module. 
Add a call slot to an existing interface slot via drag and drop (compatible call slots are highlighted if an interface slot is hovered).
Callee interface slots (on the left side of a group) allow only one connected call slot.
Interface slots are stored in project files as part of the configurators state parameter.                       

### Hotkey Overview

The following hotkeys are used in the GUI plugin:

**Global *(in class GUIWindows)*:**
* Trigger Screenshot:        **`F2`**
* Toggle Graph Entry:        **`F3`**
* Show/hide Windows:         **`F7-F11`**
* Show/hide Menu:            **`F12`**
* Show/hide GUI:             **`Ctrl  + g`**
* Search Parameter:          **`Ctrl  + p`**
* Save Running Project:      **`Ctrl  + s`**
* Quit Program:              **`Alt   + F4`**

**Configurator *(in class Configurator)*:**
* Search Module:             **`Ctrl + Shift + m`**
* Search Parameter:          **`Ctrl + Shift + p`**
* Save Edited Project:       **`Ctrl + Shift + s`**

**Graph *(in class Graph)*:**
* Delete Graph Item:         **`Entf`**
* Selection, Drag & Drop:    **`Left Mouse Button`**
* Context Menu:              **`Right Mouse Button`**
* Zooming:                   **`Mouse Wheel`**
* Scrolling:                 **`Middle Mouse Button`**

## Modules

Additional modules provided by the GUI plugin:

### GUIView

The GUIView module is deprecated. It was required for using the GUI with the former MegaMol frontend provided by mmconsole.exe / mmconsole.sh.

### OverlayRenderer

The `OverlayRenderer` is a rendering module which provides overlay rendering like textures, text, parameter values and transport control icons. Prepend the `OverlayRenderer` module any other existing 3D renderer module you want to have an overlay for. 

#### Parameters

(The values in brackets indicate the default values.)

* `mode` (`Texture`): Overlay mode.
* `anchor` (`Left Top`): Anchor of overlay.
* `position_offset` (`0.0`): Custom relative position offset in respect to selected anchor.
* `texture::file_name` (` `): The file name of the texture.
* `texture::relative_width` (`25.0`): Relative screen space width of texture.
* `transport_ctrl::color` (`0.5f, 0.75f, 0.75f`): Color of transport ctrl icons.
* `transport_ctrl::duration` (`3.0`): Duration transport ctrl icons are shown after value changes. Value of zero means showing transport ctrl icons permanently.
* `transport_ctrl::fast_speed` (`5.0`): Define factor of default speed for fast transport ctrl icon threshold.
* `transport_ctrl::value_scaling` (`10.0`): Define factor of default speed for ultra-fast transport ctrl icon threshold.
* `transport_ctrl::speed_parameter_name` (` `): The full parameter name for the animation speed, e.g. *::Project_1::View3D_21::anim::speed*.
* `transport_ctrl::time_parameter_name` (` `): The full parameter name for the animation time, e.g. *::Project_1::View3D_21::anim::time*.
* `parameter::prefix` (` `): The parameter value prefix.
* `parameter::sufix` (` `): The parameter value suffix.
* `parameter::name` (` `): The full parameter name, e.g. *::Project_1::View3D_21::cam::position*. Supported parameter types: float, int, Vector2f/3f/4f.
* `label::text` (` `): The displayed text.
* `font::name` (`Roboto Sans`): The font name.
* `font::size` (`20.0`): The font size.
* `font::color` (`0.5f, 0.5f, 0.5f`): The font color.

---

## Information for Developers

### Using ImGui in Modules

If you want to use ImGui in your module (in another plugin) you just have to apply the following steps:

**1)** Add imgui as external dependency to your plugin's `CMakeLists.txt`:
```cmake
require_external(imgui)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui)
```
**2)** Include the following ImGui headers in your modules's header:
```c++
#include "imgui.h"
#include "imgui_internal.h"
```
**3)** Before calling any ImGui function check for active ImGui context and valid scope:
```c++
bool valid_imgui_scope = ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));
if (!valid_imgui_scope) return;
```
**4)** Then call your own ImGui stuff... See [https://github.com/ocornut/imgui](https://github.com/ocornut/imgui)

### New Parameter Widgets

Parameter widgets can be defined for the `local` or the `global` scope. The widgets of the local scope are shown inside the parameter list in the `All Parameters` window. The global scope of a parameter widget can contain arbitrary ImGui code.
There are widgets for the basic parameter types defined in `gui/src/graph/Parameter.h`and there are group widgets bundling parameters of the same namespace in `gui/src/graph/ParameterGroups.h`. The parameters namespace is a prefix in the parameters, name delimited with `::`. The presentation of a parameter (group) can be changed by switching to `expert` mode in the `All Parameters` window an clicking on the 'circle button'.

#### How to add a new parameter widget

In order to add a new custom widget for a basic parameter type, the following steps are required:

* Add a new `Presentation` to `core/include/mmcore/param/AbstractParamPresentation.h` and add a human readable name to the `presentation_name_map` in the ctor.
* Add the new `Presentation` to the `compatible` presentation of a specific `ParamType` in `AbstractParamPresentation::InitPresentation()`.
* Add a new function implementing the new widget in `local` and/or `global` scope to the class `ParameterPresentation` of the pattern `widget_<NEW_WIDGET_NAME>(...)`.
    This function must return true if the parameters value has been modified!
* Register this new function in the parameter type map in `ParameterPresentation::present_parameter()` for the respective parameter type(s) - for details see existing code.

#### How to add a new parameter group widget

In order to add a new custom widget for a group of parameters sharing the same namespace within a module, the following steps are required:

* Add a new `ParamType` for new widget group in `core/include/mmcore/param/AbstractParamPresentation.h`.
* Add a new `Presentation` for new widget group in `core/include/mmcore/param/AbstractParamPresentation.h` and add a human readable name to the `presentation_name_map` in the ctor.
* Add a new case for the `ParamType`  in `AbstractParamPresentation::InitPresentation()` and add the new `Presentation` to its `compatible` presentation.
* Add a new function implementing the new group widget in `LOCAL` and/or `GLOBAL` scope to the class `ParameterGroups`. 
The function should be named like: `void draw_group_widget_<NEW_WIDGET_NAME>(...)`. See existing functions for the required signature.
* Add a new function checking for the expected and required parameters. 
The function should be named like: `bool check_group_widget_<NEW_WIDGET_NAME>(..)`. See existing functions for the required signature.
* Add a new group widget data set of the type `GroupWidgetData` in the ctor and register above functions as callbacks. 
* Identification of parameter widget groups. Currently by name of namespace and name and type of expected parameters.

### Default GUI State 

This is the default GUI state stored as JSON string in the lua project file:

```lua
mmSetGUIVisible(true)
mmSetGUIScale(1.000000)
mmSetGUIState([=[{"ConfiguratorState":{"module_list_sidebar_width":250.0,"show_module_list_sidebar":true},"GUIState":{"font_file_name":"","font_size":13,"imgui_settings":"[Window][DockSpaceViewport_11111111]\nPos=0,18\nSize=1280,702\nCollapsed=0\n\n[Window][Debug##Default]\nPos=60,60\nSize=400,400\nCollapsed=0\n\n[Window][Parameters     F10]\nPos=0,18\nSize=400,500\nCollapsed=0\n\n[Window][Log Console     F9]\nPos=0,544\nSize=1280,176\nCollapsed=0\n\n[Window][Save Project (.lua)]\nPos=440,110\nSize=400,500\nCollapsed=0\n\n[Docking][Data]\nDockSpace ID=0x8B93E3BD Window=0xA787BDB4 Pos=0,18 Size=1280,702 CentralNode=1\n\n","menu_visible":true,"style":2},"GraphStates":{"Project":{"canvas_scrolling":[0.0,0.0],"canvas_zooming":1.0,"param_extended_mode":false,"parameter_sidebar_width":300.0,"params_readonly":false,"params_visible":true,"project_name":"Project_1","show_call_label":true,"show_call_slots_label":false,"show_grid":false,"show_module_label":true,"show_parameter_sidebar":true,"show_slot_label":false}},"WindowConfigurations":{"Configurator":{"fpsms_max_value_count":20,"fpsms_mode":0,"fpsms_refresh_rate":2.0,"fpsms_show_options":false,"log_force_open":true,"log_level":4294967295,"param_extended_mode":false,"param_module_filter":0,"param_modules_list":[],"param_show_hotkeys":false,"tfe_active_param":"","tfe_view_minimized":false,"tfe_view_vertical":false,"win_callback":6,"win_collapsed":false,"win_flags":1032,"win_hotkey":[300,0],"win_position":[0.0,0.0],"win_reset_position":[0.0,0.0],"win_reset_size":[1280.0,720.0],"win_show":false,"win_size":[1280.0,720.0]},"Log Console":{"fpsms_max_value_count":20,"fpsms_mode":0,"fpsms_refresh_rate":2.0,"fpsms_show_options":false,"log_force_open":true,"log_level":4294967295,"param_extended_mode":false,"param_module_filter":0,"param_modules_list":[],"param_show_hotkeys":false,"tfe_active_param":"","tfe_view_minimized":false,"tfe_view_vertical":false,"win_callback":7,"win_collapsed":false,"win_flags":3072,"win_hotkey":[298,0],"win_position":[0.0,544.0],"win_reset_position":[0.0,544.0],"win_reset_size":[1280.0,176.0],"win_show":true,"win_size":[1280.0,176.0]},"Parameters":{"fpsms_max_value_count":20,"fpsms_mode":0,"fpsms_refresh_rate":2.0,"fpsms_show_options":false,"log_force_open":true,"log_level":4294967295,"param_extended_mode":false,"param_module_filter":0,"param_modules_list":[],"param_show_hotkeys":false,"tfe_active_param":"","tfe_view_minimized":false,"tfe_view_vertical":false,"win_callback":1,"win_collapsed":false,"win_flags":8,"win_hotkey":[299,0],"win_position":[0.0,18.0],"win_reset_position":[0.0,0.0],"win_reset_size":[400.0,500.0],"win_show":true,"win_size":[400.0,500.0]},"Performance Metrics":{"fpsms_max_value_count":20,"fpsms_mode":0,"fpsms_refresh_rate":2.0,"fpsms_show_options":false,"log_force_open":true,"log_level":4294967295,"param_extended_mode":false,"param_module_filter":0,"param_modules_list":[],"param_show_hotkeys":false,"tfe_active_param":"","tfe_view_minimized":false,"tfe_view_vertical":false,"win_callback":3,"win_collapsed":false,"win_flags":65,"win_hotkey":[296,0],"win_position":[640.0,0.0],"win_reset_position":[640.0,0.0],"win_reset_size":[0.0,0.0],"win_show":false,"win_size":[0.0,0.0]},"Transfer Function Editor":{"fpsms_max_value_count":20,"fpsms_mode":0,"fpsms_refresh_rate":2.0,"fpsms_show_options":false,"log_force_open":true,"log_level":4294967295,"param_extended_mode":false,"param_module_filter":0,"param_modules_list":[],"param_show_hotkeys":false,"tfe_active_param":"","tfe_view_minimized":false,"tfe_view_vertical":false,"win_callback":5,"win_collapsed":false,"win_flags":64,"win_hotkey":[297,0],"win_position":[400.0,0.0],"win_reset_position":[400.0,0.0],"win_reset_size":[0.0,0.0],"win_show":false,"win_size":[0.0,0.0]}}}]=])
```

### Graph Data Structure

![gui graph structure](graph_structure.png)

### Class Dependencies

![gui plugin class dependencies](class_dependencies.png)
