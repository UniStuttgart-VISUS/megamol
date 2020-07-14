/*
 * OpenGL_Context.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol {
namespace input_events {

struct IOpenGL_Context {
	virtual void activate() const = 0;
    virtual void close() const = 0;

	virtual ~IOpenGL_Context() = default;
};

} /* end namespace input_events */
} /* end namespace megamol */
