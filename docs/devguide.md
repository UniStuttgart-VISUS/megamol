# MegaMol Developer Guide

This guide is intended to give MegaMol developers a useful insight into the internal structure of MegaMol.

<!-- TOC -->

## Contents

- [MegaMol Developer Guide](#megamol-developer-guide)
  - [Contents](#contents)
  - [Create new Plugin](#create-new-plugin)
    - [Add own plugin using the template](#add-own-plugin-using-the-template)
  - [Create GLSL Shader with utility classes](#create-glsl-shader-with-utility-classes)
  - [Bi-Directional Communication across Modules](#bi-directional-communication-across-modules)
    - [Definitions](#definitions)
    - [Recipe](#recipe)
      - [Interaction with the ```DATA``` from outside:](#interaction-with-the-data-from-outside)
      - [*owner* Module:](#owner-module)
  - [Synchronized Selection across Modules](#synchronized-selection-across-modules)
    - [FlagStorage](#flagstorage)
    - [FlagStorage_GL](#flagstorage_gl)
  - [1D Transfer Function](#1d-transfer-function)
    - [Usage](#usage)
  - [Graph Manipulation](#graph-manipulation)
    - [Graph Manipulation Queues](#graph-manipulation-queues)
  - [Build System](#build-system)
    - [External dependencies](#external-dependencies)
      - [Using external dependencies](#using-external-dependencies)
      - [Adding new external dependencies](#adding-new-external-dependencies)
        - [Header-only libraries](#header-only-libraries)
        - [Built libraries](#built-libraries)
  - [GUI](#gui)
    - [Parameter Widgets](#parameter-widgets)
    - [Window/PopUp/Notification for Frontend Service](#windowpopupnotification-for-frontend-service)

<!-- TODO
- Add section describing all available LUA commands
- Add section describing remote console usage
-->

<!-- /TOC -->


<!-- ###################################################################### -->
-----
## Create new Plugin

### Add own plugin using the template

1. Copy the template folder `plugins/doc_template`.
2. Rename the copied folder to the intended plugin name (style guide: only lower case letters, numbers, and underscore).
3. Rename the `src/megamolplugin.cpp` to your plugin name (style guide: same as folder name). Within the file change the following:
    1. Use a unique namespace `megamol::pluginname`. (style guide: same as folder name)
    2. Change the plugin name and description in the parameters of the constructor.
    3. The class name can be changed to any name, but it must be set accordingly in the `REGISTERPLUGIN()` macro.
4. Open the `CMakeLists.txt` file and to the following changes:
    1. Set the name of the target at the beginning of `megamol_plugin()`. (style guide: same as folder name)
    2. List the required features of the plugin after `DEPENDS_FEATURES`.
    3. List the targets of other plugin dependencies after `DEPENDS_PLUGINS`[*].
    4. Add any dependencies or additional CMake configuration within `if (megamolplugin_PLUGIN_ENABLED)`.
       Dependencies are defined in the regular vcpkg way (`find_package()`, `target_link_libraries()`, etc., see [external dependencies](#external-dependencies)).
       The variable defined at the beginning of `megamol_plugin()` is a regular CMake target that can be used.
5. Implement the content of your plugin.
    1. The private implementation should be in the `<pluginname>/src` directory. Source files are added automatically within CMake.
    2. If the plugin has a public interface, add the headers in the `<pluginname>/include` directory (set visibility of dependencies accordingly, see [*]).
    3. If the plugin uses shaders, add them into the `<pluginname>/shaders/<pluginname>` directory (see shader guide for more details).
    4. If the plugin uses resources, add them to `<pluginname>/resources`.
6. Write a `README.md` for your plugin (mandatory).

[*] You can prefix the dependency targets with the keywords `PUBLIC`, `PRIVATE`, or `INTERFACE` the same way `target_link_libraries()` works. Defaults to `PRIVATE` if nothing is set.

<!-- ###################################################################### -->
-----

## Create GLSL Shader with utility classes

### Shader files

Shaders are stored as regular text files containing GLSL code, but we additionally are using a shader factory that supports and include directives similar to C/C++.
This allows a better organization of large shaders and reusing of common shader snippets.

Shader files must be located either in the `core/shaders/core` directory or in the shader directory of the corresponding plugin `<pluginname>/shaders/<pluginname>`.
Please note the additional subfolder within each `shaders` directory!
It is required for our include system and to avoid collision when installing all shaders to a single shader-installation-directory.

Full shaders must use the filename pattern `<path>/<name>.<type>.glsl`.
We use the filename to determine the type of a shader.
The following types are allowed:

| Type | Shader Stage            |
| ---- | ----------------------- |
| vert | Vertex                  |
| tesc | Tessellation Control    |
| tese | Tessellation Evaluation |
| geom | Geometry                |
| frag | Fragment                |
| comp | Compute                 |

Shader snippets meant for inclusion within other shaders must not use a type in their extension, but they should still have a `.glsl` extension.

Within shader files, includes can be defined in standard C style: `#include "common.h"`.
Thereby, `core/shaders` and all `<pluginname>/shaders` directories are added as include search paths.
Therefore, shaders can be included with a path relative to these directories. (This is also the reason why we need the additional subdirectories as mentioned above).
Some examples:
```glsl
#include "core/phong.glsl"
#include "pluginname/foo/bar.glsl"
```

In addition, inclusion relative to the current shader file is supported, but only for snippets within the same shader directory.

### C++ Code

Required headers:
- `mmcore/CoreInstance.h`
- `mmcore/utility/ShaderFactory.h`

Before creating a shader program with this wrapper, `compiler_options` need to be retrieved from the `CoreInstance`.
This `compiler_options` instance contains default shader paths and default options.
Additional include paths and definitions can be added before program creation.
The constructor of the wrapper requires paths to source files of all shader stages in the form: `<path>/<name>.<type>.glsl`.

Here is a full example:
```cpp
const auto shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
auto program = core::utility::make_glowl_shader("name", shader_options, "pluginname/shader.comp.glsl");
```

The `make_glowl_shader()` helper function returns a glowl `GLSLProgram` wrapped in a unique pointer.
A variant `make_shared_glowl_shader()` exists to use a shared pointer instead.
Both functions are variadic, so any number of shaders can be combined into a single program (as long as the shader combination makes sense from the OpenGL perspective).

## Bi-Directional Communication across Modules

Bi-Directional Communication, while in principle enabled on any Call in MegaMol, has severe implications when data can be changed at an arbitrary end and multiple Modules work on the data that is passed around. The current state of knowledge is described here and should be used for any and all Calls moving forward.
You can have a look for an example in the ```FlagStorage_GL``` together with the ```FlagCall_GL``` and ```GlyphRenderer``` (among others).

We refer to the data passed around simply as ```DATA```, you can think of this as a struct containing everything changed by the corresponding Call.
To properly track changes across several Modules, you need to follow the recipe.

### Definitions
- There is one Module holding and owning the ```DATA```. It is also responsible for resizing it (e.g. in occasion of an incoming update) and disposing it (on destruction). This is the *owner*. It can have a notion of what it is holding and manipulate the ```DATA```, but this is not necessary. It can also just be a container that is talked to by other modules.
- There can be several Modules that read the ```DATA```. These are *consumer*s.
- There can be several Modules that write/update the ```DATA```. These are *supplier*s.
- Note that Modules that are *supplier*s automatically have *consumer* status as well, otherwise updates are lost.
### Recipe
Split up the data flow for each direction, one call for reading only, one call for writing only.
Keep in mind that the caller by definition is "left" in the module graph and the callee is "right". The callee is the end of a callback, but for this pattern this has nothing to do with the direction the data flows in, which results in the rules defined below.

![bidir example](images/bidir.png)

- set up a ```uint32_t version_DATA``` in the *owner*
- Create a ```DATACallRead``` either instancing the ```core::GenericVersionedCall``` template, or making sure that the call can distinguish between the ```DATA``` version that was last **set** into the Call and that which was last **got** out of the Call
- Create a ```DATACallWrite``` along the same lines
- create a ```CalleeSlot``` each for ```DATACallRead``` and ```DATACallWrite``` for the *owner*
- create a ```CallerSlot``` for ```DATACallRead``` for *consumer*s
- create a ```CallerSlot``` for ```DATACallWrite``` for *supplier*s

#### Interaction with the ```DATA``` from outside:
This must always follow the same order to avoid inconsistencies.

The following 'block' must be executed without the control flow leaving the current module, i.e. the only calls that are executed are the ones connected to the owner module that are part of the bidirectional flow.
- if *consumer*: execute the reading:
  - issue the Call ```(*call)(DATACallRead::CallGetData)``` or your specialized version
  - check if something new is available: ```call::hasUpdate()```
  - overwrite/replace what notion you had of ```DATA``` if there is an incoming update
  - take note of the ```call::version()``` *V* of the incoming update.
- if (also) *supplier*:
  - if you need to modify the ```DATA``` (parameters, other incoming data, or user input cause this, for example)
    - write the update
    - increase *V*
  - **ALWAYS** set the ```DATA``` in the ```DATACallWrite```, supplying *V*
  - **ALWAYS** issue the Call: ```(*call)(DATACallWrite::CallGetData)``` or your specialized version

Since this should take place in the same callback, *V* **must** not be kept around beyond that, it must be re-fetched in the next cycle anyway: any calls that go downstream before or after the above block could potentially alter the ```DATA``` in the *owner*.

#### *owner* Module:

Usage: ```DATACallRead```
- In the ```GetData``` callback, make sure you can **supply unchanged data very cheaply**.
  - If parameters or incoming data (downstream) have changed, modify your ```DATA``` accordingly, **increasing** ```version_DATA```.
  - **ALWAYS** set the data in the call, supplying the version.
  
Usage: ```DataCallWrite```
- In the ```GetData``` callback
  - If the incoming set data is newer than what you last got: ```call::hasUpdate()```
    - Fetch the data
    - fast-forward version to the available one ```version_DATA = call::version()```


<!-- ###################################################################### -->
-----
## Synchronized Selection across Modules

You can and should use one of the ```FlagStorage``` variants to handle selection. These modules provide an array of ```uint32_t``` that is implicitly index-synchronized (think row, record, item) to the data available to a renderer. Indices are accumulated across primitives and primitive groups, so if you employ these, you need to take care yourself that you always handle Sum_Items flags and apply proper **offsets** across, e.g., particle lists.

### FlagStorage

The flags here are stored uniquely, resembling a unique pointer or Rust-like memory handover. A unique pointer still cannot be used with the current Call mechanisms (specifically, the leftCall = rightCall paradigm). So, unlike any other modules, asking for Flags via ```CallMapFlags``` will **remove** the flags from the FlagStorage and place them in the ```FlagCall```. In the same way, if you ```GetFlags```, you will own them and **need to give them back** when finished operating on them (```SetFlags```, then ```CallUnmapFlags```). Any other way of using them will **crash** other modules that operate on the same flags (meaning the FlagStorage tries to tell you that you are using them inappropriately).

***TODO: this still requires the new Bi-Directional Communication paradigm.***

### FlagStorage_GL

This FlagStorage variant relies on a shader storage buffer and does not move any data around. It is implicitly synchronized by single-threaded execution in OpenGL. You still need to synchronize with the host if you want to download the data though. It still keeps track of proper versions so you can detect and act on changes, for example when synchronizing a FlagStorage and a FlagStorage_GL.


<!-- ###################################################################### -->
-----
## 1D Transfer Function

***... TODO ...***

<!--

The whole functionality of a 1D transfer function is provided via the module `TransferFunction` which holds the actual `TransferFunctionParam` parameter.

If you want to use a transfer function in you renderer module you have to create a caller slot, which is compatible to the call `CallGetTransferFunction`:
```C++
    this->tfSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->tfSlot);
```

The description of the transfer function is encoded in a string in JSON format, see header `TransferFunctionParam.h`.

-->
### Usage

***... TODO ...***

<!--
See the header file of the call `CallGetTransferFunction` for a more detailed interface description of the available functions.
The renderer modules `SimpleSphereRenderer` or `ScatterplotMatrixRenderer2D`can be looked at for a example implementation of the transfer function.

-->

<!-- ###################################################################### -->
-----
## Graph Manipulation

There are appropriate methods in the ```megamol::core::CoreInstance``` to traverse, search, and manipulate the graph.
**Locking the graph** is only required for code that runs **concurrently**.
At this point, MegaMol graph execution happens sequentially, so any Module code can only run concurrently when you split off a thread yourself.
Services (children of ```megamol::core::AbstractService```), on the other hand, always run concurrently, so they need to lock the graph.
All graph manipulation needs to be requested and is buffered, as described in the following section.

### Graph Manipulation Queues

Graph manipulation requests are queued and executed between two frames in the main thread.
There are different queues for different types of requests:

| Name                         | Description                                                | Entry Type                               |
| ---------------------------- | ---------------------------------------------------------- | ---------------------------------------- |
| pendingViewInstRequests      | Views to be instantiated                                   | ViewInstanceRequest                      |
| pendingJobInstRequests       | Jobs to be instantiated                                    | JobInstanceRequest                       |
| pendingCallInstRequests      | Requests to instantiate calls (from, to)                   | CallInstanceRequest                      |
| pendingChainCallInstRequests | Requests to instantiate chain calls (from chain start, to) | CallInstanceRequest                      |
| pendingModuleInstRequests    | Modules to be instantiated                                 | ModuleInstanceRequest                    |
| pendingCallDelRequests       | Calls to be deleted                                        | ASCII string (from), ASCII string (to)   |
| pendingModuleDelRequests     | Modules to be deleted                                      | ASCII string (id)                        |
| pendingParamSetRequests      | Requests to set parameters                                 | Pair of ASCII strings (parameter, value) |
| pendingGroupParamSetRequests | Requests to create parameter group                         | Pair of ASCII string (id) and ParamGroup |

For each of this queues, there is a list with indices into the respective queue pointing to the last queued event before a flush.
It causes the graph updater to stop at the indicated event and delay further graph updates to the next frame.

| Name                              |
| --------------------------------- |
| viewInstRequestsFlushIndices      |
| jobInstRequestsFlushIndices       |
| callInstRequestsFlushIndices      |
| chainCallInstRequestsFlushIndices |
| moduleInstRequestsFlushIndices    |
| callDelRequestsFlushIndices       |
| moduleDelRequestsFlushIndices     |
| paramSetRequestsFlushIndices      |
| groupParamSetRequestsFlushIndices |


<!-- ###################################################################### -->
-----
## Module Behavior

A module should always try to do its work, even with missing resources.
Examples:
- All views should clear themselves to their internal background color, even if no other module is connected.

TODO: Module behavior in case of missing resources or call connections.

<!-- ###################################################################### -->
-----
## Build System

For building MegaMol, CMake is used. For developers, two aspects are of importance: [adding new plugins](#create-new-plugin), and [adding and using external dependencies](#external-dependencies).

### External dependencies

We are using [vcpkg](https://github.com/microsoft/vcpkg) (in manifest mode) for managing external dependencies.
In short, vcpkg will automatically download, build and cache any dependency during configuring MegaMol.
After that common CMake methods are used to use the library (i.e. `find_package()` and `target_link_libraries()`, but of course depends on the external library).

For dependencies already available within the vcpkg ecosystem, just add them to `vcpkg.json`.
For dependencies not already available, an overlay port must be created within `cmake/vcpkg_ports/<port-name>`.
Further information on writing new ports is available in the [vcpkg documentation](https://vcpkg.io/en/docs/README.html).

Larger (especially in the meaning of build times) and/or very specialized libraries could be wrapped within vcpkg features in `vcpkg.json`.
If doing so add an option to the beginning of `CMakeLists.txt` using `megamol_feature_option()` to allow using that feature.

#### Overriding dependencies

If you need customized veriants of a dependency, say, an OSPRay build with custom modules, you can switch to a version that lives somewhere else on your system. For this, you can (temporarily) create an additional port with the same name in cmake/vcpkg_ports. **NEVER COMMIT THIS PORT!**

The port needs to contain a `vcpkg.json` that includes all relevant dependencies and features that are requested by megamol. You can just use the file from the port you are overriding. You also need to place a *completely empty* `portfile.cmake` in the port directory.

You are responsible for building the custom version in the correct way yourself. A custom drop-in replacement OSPRay that re-uses all the dependencies that are built with MegaMol anyway would, for example, be configured using options along the lines of:

```-Drkcommon_DIR:PATH="drive:/path/to/megamol/megamol/build/vs-ninja-22/vcpkg_installed/x64-windows/share/rkcommon" -DISPC_EXECUTABLE:FILEPATH="drive:/path/to/megamol/megamol/build/vs-ninja-22/vcpkg_installed/x64-windows/tools/ispc/ispc.exe" -DOSPRAY_ENABLE_APPS_EXAMPLES:BOOL="0" -DTBB_DIR:PATH="" -DTBB_ROOT:PATH="drive:/path/to/megamol/megamol/build/vs-ninja-22/vcpkg_installed/x64-windows" -Dembree_DIR:PATH="drive:/path/to/megamol/megamol/build/vs-ninja-22/vcpkg_installed/x64-windows/share/embree" -DOSPRAY_ENABLE_APPS_TESTING:BOOL="0" -Dopenvkl_DIR:PATH="drive:/path/to/megamol/megamol/build/vs-ninja-22/vcpkg_installed/x64-windows/share/openvkl" -DOSPRAY_ENABLE_APPS_BENCHMARK:BOOL="0" -DOSPRAY_ENABLE_APPS_TUTORIALS:BOOL="0"```

Do not use the OSPRay superbuild as it includes and references its own set of transitive dependencies, which could mean that you end up with two different versions of TBB, for example. It also pollutes CMake with additional targets that confuse vcpkg.

To make sure the megamol build finds the custom dependency build, you can set the corresponding environment variable manually or in your `CMakeUserPresets.json`. Keeping with the previous example, this could read `"environment": {"OSPRAY_ROOT": "drive:/some/directory/ospray/install"}`. You can then either copy the resulting dlls manually to the binary directory or TODO.

<!-- ###################################################################### -->
-----
## GUI

### Parameter Widgets

See [developer information for GUI Service](../../frontend/services/gui#new-parameter-widgets).

### Window/PopUp/Notification for other Frontend Services

See [developer information for GUI Service](../../frontend/services/gui#gui-windowpopupnotification-for-frontend-service).
