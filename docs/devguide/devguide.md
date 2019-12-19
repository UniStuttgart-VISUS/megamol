# MegaMol Developer Guide

<!-- TOC -->

## Contents

- [MegaMol Developer Guide](#megamol-developer-guide)
    - [Contents](#contents)
    - [Overview](#overview)
    - [Graph Manipulation Queues](#graph-manipulation-queues)
- [License](#license)

<!-- /TOC -->

## Overview

This guide is intended to give MegaMol developers a useful insight into the internal structure of MegaMol.

# Bi-Directional Communication across Modules

Bi-Directional Communication, while in principle enabled on any Call in MegaMol, has severe implications when data can be changed at an arbitrary end and multiple Modules work on the data that is passed around. The current state of knowledge is described here and should be used for any and all Calls moving forward.
You can have a look for an example in the ```FlagStorage_GL``` together with the ```FlagCall_GL``` and ```GlyphRenderer``` (among others).

We refer to the data passed around simply as ```DATA```, you can think of this as a struct containing everything changed by the corresponding Call.
To properly track changes across several Modules, you need to follow the recipe.

## Recipe
Split up the data flow for each direction, one call for reading only, one call for writing only.
Keep in mind that the caller by definition is "left" in the module graph and the callee is "right". The callee is the end of a callback, but for this pattern this has nothing to do with the direction the data flows in, which results in the rules defined below.
- set up a ```uint32_t version_DATA``` in each Module
- create a ```CallerSlot``` for ```DATACallRead``` for modules reading the ```DATA```
- (optional) create a ```CallerSlot``` for ```DATACallWrite``` for modules writing the ```DATA```
- Create a ```DATACallRead``` either instancing the ```core::GenericVersionedCall``` template, or making sure that the call can distinguish between the ```DATA``` version that was last **set** into the Call and that which was last **got** out of the Call
- (optional) Create a ```DATACallWrite``` along the same lines
- create a ```CalleeSlot``` for ```DATACallRead``` for modules consuming the ```DATA```
- create a ```CallerSlot``` for ```DATACallWrite``` for modules supplying the ```DATA```

A module with slots for both directions by convention should first execute the reading and then provide updates via writing.

### Usage: ```DATACallRead```
- In the ```GetData``` callback, make sure you can *supply unchanged data very cheaply*.
  - If parameters or incoming data have changed, modify your ```DATA``` accordingly, **increasing** your version.
  - **ALWAYS** set the data in the call, supplying your version.
- As a consumer
    - issue the Call: ```(*call)(DATACallRead::CallGetData)``` or your specialized version
    - check if something new is available: ```call::hasUpdate()```
    - if necessary, fast-forward your own version to the available one ```version_DATA = GenericVersionedCall::version()```

### Usage: ```DataCallWrite```
- In the ```GetData``` callback
  - First check if the incoming set data is newer than what you last got: ```GenericVersionedCall::hasUpdate()```
  - Fetch the data and if necessary, fast-forward your own version to the available one ```version_DATA = GenericVersionedCall::version()```
- As a provider
  - If parameters or incoming data have changed, modify your ```DATA``` accordingly, **increasing** your version.
  - **ALWAYS** set the data in the call, supplying your version.
  - issue the Call: ```(*call)(DATACallRead::CallGetData)``` or your specialized version

# Synchronized Selection across Modules

You can and should use one of the ```FlagStorage``` variants to handle selection. These modules provide an array of ```uint32_t``` that is implicitly index-synchronized (think row, record, item) to the data available to a renderer. Indices are accumulated across primitives and primitive groups, so if you employ these, you need to take care yourself that you always handle Sum_Items flags and apply proper **offsets** across, e.g., particle lists.

## FlagStorage

The flags here are stored uniquely, resembling a unique pointer or Rust-like memory handover. A unique pointer still cannot be used with the current Call mechanisms (specifically, the leftCall = rightCall paradigm). So, unlike any other modules, asking for Flags via ```CallMapFlags``` will **remove** the flags from the FlagStorage and place them in the ```FlagCall```. In the same way, if you ```GetFlags```, you will own them and **need to give them back** when finished operating on them (```SetFlags```, then ```CallUnmapFlags```). Any other way of using them will **crash** other modules that operate on the same flags (meaning the FlagStorage tries to tell you that you are using them inappropriately).

***TODO: this still requires the new Bi-Directional Communication paradigm.***

## FlagStorage_GL

This FlagStorage variant relies on a shader storage buffer and does not move any data around. It is implicitly synchronized by single-threaded execution in OpenGL. You still need to synchronize with the host if you want to download the data though. It still keeps track of proper versions so you can detect and act on changes, for example when synchronizing a FlagStorage and a FlagStorage_GL.

# Graph Manipulation

There are appropriate methods in the ```megamol::core::CoreInstance``` to traverse, search, and manipulate the graph.
**Locking the graph** is only required for code that runs **concurrently**.
At this point, MegaMol graph execution happens sequentially, so any Module code can only run concurrently when you split off a thread yourself.
Services (children of ```megamol::core::AbstractService```), on the other hand, always run concurrently, so they need to lock the graph.
All graph manipulation needs to be requested and is buffered, as described in the following section.

## Graph Manipulation Queues

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

|Name|
|---|
|viewInstRequestsFlushIndices|
|jobInstRequestsFlushIndices|
|callInstRequestsFlushIndices|
|chainCallInstRequestsFlushIndices|
|moduleInstRequestsFlushIndices|
|callDelRequestsFlushIndices|
|moduleDelRequestsFlushIndices|
|paramSetRequestsFlushIndices|
|groupParamSetRequestsFlushIndices|


## License

MegaMol is freely and publicly available as open source following the therms of the BSD License.
Copyright (c) 2015, MegaMol Team TU Dresden, Germany Visualization Research Center, University of Stuttgart (VISUS), Germany
Alle Rechte vorbehalten.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the MegaMol Team, TU Dresden, University of Stuttgart, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE MEGAMOL TEAM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE MEGAMOL TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
