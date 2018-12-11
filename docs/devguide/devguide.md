# MegaMol Developer Guide

<!-- TOC -->

## Contents

- [MegaMol Developer Guide](#megamol-developer-guide)
    - [Contents](#contents)
    - [Overview](#overview)
        - [License](#license)
    - [Graph Manipulation Queues](#graph-manipulation-queues)

<!-- /TOC -->

## Overview

This guide is intended to give MegaMol developers a useful insight into the internal structure of MegaMol.

### License

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
