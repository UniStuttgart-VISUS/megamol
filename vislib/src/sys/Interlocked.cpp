/*
 * Interlocked.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "vislib/sys/Interlocked.h"

#include "vislib/UnsupportedOperationException.h"
#include "vislib/assert.h"


/*
 * vislib::sys::Interlocked::~Interlocked
 */
vislib::sys::Interlocked::~Interlocked() {
    // Should never be called.
    ASSERT(false);
}


/*
 * vislib::sys::Interlocked::Interlocked
 */
vislib::sys::Interlocked::Interlocked() {
    throw UnsupportedOperationException("Interlocked", __FILE__, __LINE__);
}
