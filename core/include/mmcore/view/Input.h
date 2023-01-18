/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "KeyboardMouseInput.h"
// TODO: do this include correctly via CMake

namespace megamol::core::view {

using namespace megamol::frontend_resources;
// the definitions for Keyboard and Mouse inputs moved to a seperate CMake submodule.
// to not break old code we import the definitions here and make them also available via this old interface.
// TODO: refactor old code to use new KeyboardMouseInput.h file from frontend_resources CMake module
// the new file for the old code is at: frontend_resources/inlude/KeyboardMouseInput.h

} // namespace megamol::core::view
