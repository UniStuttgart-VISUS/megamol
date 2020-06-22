/*
 * Input.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <KeyboardMouseInput.h>
// TODO: do this include correctly via CMake

namespace megamol {
namespace core {
namespace view {

	using namespace megamol::input_events;
	// the definitions for Keyboard and Mouse inputs moved to a seperate CMake submodule.
	// to not break old code we import the definitions here and make them also available via this old interface.
	// TODO: refactor old code to use new KeyboardMouseInput.h file from input_events CMake module
	// the new file for the old code is at: input_events/inlude/KeyboardMouseInput.h

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
